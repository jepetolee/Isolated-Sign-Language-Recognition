import json
import math
from pathlib import Path

import numpy as np
import onnx
import pandas as pd
import pytorch_lightning as pl
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import wandb
from onnx_tf.backend import prepare
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import StratifiedGroupKFold
from timm.optim import create_optimizer_v2
from torchmetrics import MetricCollection

KAGGLE_DIR = Path("/") / "kaggle"
INPUT_DIR = KAGGLE_DIR / "input"
DATA_ROOT_DIR = INPUT_DIR / "asl-signs"
NPY_DATA_DIR = INPUT_DIR / "asl-signs-features-npy"

TRAIN_CSV_PATH = NPY_DATA_DIR / "train_prepared.csv"
SIGN_TO_IDX_PATH = INPUT_DIR / "asl-signs" / "sign_to_prediction_index_map.json"

with open(SIGN_TO_IDX_PATH, "r") as f:
    SIGN_TO_IDX = json.load(f)

N_SPLITS = 5
SEED = 2023
ROWS_PER_FRAME = 543

IN_FEATURES = 708

df = pd.read_csv(TRAIN_CSV_PATH)


class FeatureGen(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

        self.idx_range_face = (0, 468)
        self.idx_range_hand_left = (468, 489)
        self.idx_range_pose = (489, 522)
        self.idx_range_hand_right = (522, 543)

        self.dims = 3

        # https://www.kaggle.com/competitions/asl-signs/discussion/391812#2168354
        lips_upper_outer = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
        lips_lower_outer = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
        lips_upper_inner = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
        lips_lower_inner = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
        self.lips = (
            lips_upper_outer + lips_lower_outer + lips_upper_inner + lips_lower_inner
        )

    def forward(self, x):
        x = x[:, :, : self.dims]

        x_face = self.get_flat_features(x, self.idx_range_face)
        x_lips = x[:, self.lips, :].contiguous().view(-1, len(self.lips) * self.dims)
        x_hand_left = self.get_flat_features(x, self.idx_range_hand_left)
        x_pose = self.get_flat_features(x, self.idx_range_pose)
        x_hand_right = self.get_flat_features(x, self.idx_range_hand_right)

        x_hand_left = x_hand_left[~torch.any(torch.isnan(x_hand_left), dim=1), :]
        x_hand_right = x_hand_right[~torch.any(torch.isnan(x_hand_right), dim=1), :]

        x_face_mean = torch.mean(x_face, 0)
        x_lips_mean = torch.mean(x_lips, 0)
        x_hand_left_mean = torch.mean(x_hand_left, 0)
        x_pose_mean = torch.mean(x_pose, 0)
        x_hand_right_mean = torch.mean(x_hand_right, 0)

        x_face_std = torch.std(x_face, 0)
        x_lips_std = torch.std(x_lips, 0)
        x_hand_left_std = torch.std(x_hand_left, 0)
        x_pose_std = torch.std(x_pose, 0)
        x_hand_right_std = torch.std(x_hand_right, 0)

        x_features = torch.cat(
            [
                # x_face_mean,
                x_lips_mean,
                x_hand_left_mean,
                x_pose_mean,
                x_hand_right_mean,
                # x_face_std,
                x_lips_std,
                x_hand_left_std,
                x_pose_std,
                x_hand_right_std,
            ],
            axis=0,
        )

        x_features = torch.where(
            torch.isnan(x_features), torch.tensor(0.0, dtype=torch.float32), x_features
        )

        return x_features

    def get_flat_features(self, x, idx_range):
        len_range = idx_range[1] - idx_range[0]
        return (
            x[:, idx_range[0] : idx_range[1], :]
            .contiguous()
            .view(-1, len_range * self.dims)
        )

class ASLDataFrameDataset(torch.utils.data.Dataset):
    def __init__(self, df, in_features, transform=None):
        self.df = df
        self.transform = transform

        print("Loading data...")
        self.X = np.load(NPY_DATA_DIR / f"X_{in_features}.npy")
        self.y = np.load(NPY_DATA_DIR / "y.npy")
        print(f"X shape: {self.X.shape}, y shape: {self.y.shape}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Use df_index as idx due to folds splitting
        df_index = self.df.index.values[idx]
        x = self.X[df_index]
        y = self.y[df_index]

        x = torch.Tensor(x)
        y = torch.Tensor([y]).long()

        if self.transform:
            x = self.transform(x)

        return x, y




class ASLDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        data_csv_path: str,
        in_features: int,
        num_workers: int,
        val_fold: float,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.df = pd.read_csv(data_csv_path)

        self.train_transform, self.val_transform = self._init_transforms()

    def _init_transforms(self):
        train_transform = None
        val_transform = None

        return train_transform, val_transform

    def setup(self, stage=None):
        val_fold = self.hparams.val_fold
        train_df = self.df[self.df.fold != val_fold]
        val_df = self.df[self.df.fold == val_fold]

        if stage == "fit" or stage is None:
            self.train_dataset = self._dataset(train_df, self.train_transform)
            self.val_dataset = self._dataset(val_df, self.val_transform)

    def _dataset(self, df, transform):
        return ASLDataFrameDataset(df, self.hparams.in_features, transform=transform)

    def train_dataloader(self):
        return self._dataloader(self.train_dataset, train=True)

    def val_dataloader(self):
        return self._dataloader(self.val_dataset)

    def _dataloader(self, dataset, train=False):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=train,
            num_workers=self.hparams.num_workers,
            drop_last=train,
        )

# From https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/blob/master/src/modeling/metric_learning.py
# Added type annotations, device, and 16bit support
class ArcMarginProduct(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        scale: float,
        margin: float,
        easy_margin: bool,
        ls_eps: float,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(
        self, input: torch.Tensor, label: torch.Tensor, device: str = "cuda"
    ) -> torch.Tensor:
        # --------------------------- cos(theta) & phi(theta) ---------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        # # Enable 16 bit precision
        # cosine = cosine.to(torch.float32)

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # --------------------------- convert label to one-hot ---------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) ------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        return output

class ASLLinearModel(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        first_out_features: int,
        num_classes: int,
        num_blocks: int,
        drop_rate: float,
    ):
        super().__init__()

        blocks = []
        out_features = first_out_features
        for idx in range(num_blocks):
            if idx == num_blocks - 1:
                out_features = num_classes

            blocks.append(self._make_block(in_features, out_features, drop_rate))

            in_features = out_features
            out_features = out_features // 2

        self.model = nn.Sequential(*blocks)
        print(self.model)

    def _make_block(self, in_features, out_features, drop_rate):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(drop_rate),
        )

    def forward(self, x):
        return self.model(x)


class ASLModule(pl.LightningModule):
    def __init__(
        self,
        drop_rate: float,
        eta_min: float,
        first_out_features: int,
        learning_rate: float,
        loss: str,
        in_features: int,
        max_epochs: int,
        model_name: str,
        num_blocks: int,
        num_classes: int,
        optimizer: str,
        scheduler: str,
        weight_decay: float,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model = self._init_model()

        self.loss_fn = self._init_loss_fn()

        self.metrics = self._init_metrics()

    def _init_model(self):
        if self.hparams.model_name == "linear":
            return ASLLinearModel(
                in_features=self.hparams.in_features,
                first_out_features=self.hparams.first_out_features,
                num_classes=self.hparams.num_classes,
                num_blocks=self.hparams.num_blocks,
                drop_rate=self.hparams.drop_rate,
            )
        else:
            raise ValueError(f"{self.hparams.model_name} is not a valid model name")

    def _init_loss_fn(self):
        if self.hparams.loss == "CELoss":
            return torch.nn.CrossEntropyLoss()
        else:
            raise ValueError(f"{self.hparams.loss} is not a valid loss function")

    def _init_metrics(self):
        metrics = {
            "acc": torchmetrics.classification.MulticlassAccuracy(
                num_classes=len(SIGN_TO_IDX)
            ),
        }
        metric_collection = MetricCollection(metrics)

        return torch.nn.ModuleDict(
            {
                "train_metrics": metric_collection.clone(prefix="train_"),
                "val_metrics": metric_collection.clone(prefix="val_"),
            }
        )

    def configure_optimizers(self):
        optimizer = self._init_optimizer()

        scheduler = self._init_scheduler(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def _init_optimizer(self):
        return create_optimizer_v2(
            self.parameters(),
            opt=self.hparams.optimizer,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

    def _init_scheduler(self, optimizer):
        if self.hparams.scheduler == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.max_epochs,
                eta_min=self.hparams.eta_min,
            )
        elif self.hparams.scheduler == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.hparams.max_epochs // 5,
                gamma=0.95,
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.hparams.scheduler}")
        return scheduler

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def predict_step(self, batch, batch_idx):
        _, labels, logits = self._forward_pass(batch, stage=None)
        preds = logits.sigmoid()
        return preds, labels

    def _shared_step(self, batch, stage):
        x, y, y_pred = self._forward_pass(batch)

        loss = self.loss_fn(y_pred, y)

        self.metrics[f"{stage}_metrics"](y_pred, y)

        self._log(stage, loss, batch_size=len(x))

        return loss

    def _forward_pass(self, batch):
        x, y = batch
        y = y.view(-1)
        y_pred = self(x)

        return x, y, y_pred

    def _log(self, stage, loss, batch_size):
        self.log(f"{stage}_loss", loss, batch_size=batch_size)
        self.log_dict(self.metrics[f"{stage}_metrics"], batch_size=batch_size)


def load_logger_and_callbacks(
    fast_dev_run, metrics, overfit_batches, patience, project, val_fold
):
    if fast_dev_run or overfit_batches > 0:
        logger, callbacks = None, None
    else:
        logger, id_ = get_logger(metrics=metrics, project=project)
        callbacks = get_callbacks(
            id_=id_,
            mode=list(metrics.values())[0],
            monitor=list(metrics.keys())[0],
            patience=patience,
            val_fold=val_fold,
        )

    return logger, callbacks


def get_logger(metrics, project):
    logger = WandbLogger(project=project)
    id_ = logger.experiment.id

    for metric, summary in metrics.items():
        logger.experiment.define_metric(metric, summary=summary)

    return logger, id_


def get_callbacks(id_, mode, monitor, patience, val_fold):
    callbacks = [
        EarlyStopping(monitor=monitor, mode=mode, patience=patience),
        LearningRateMonitor(),
    ]

    return callbacks


def train(
    accelerator: str = "gpu",
    batch_size: int = 256,
    devices: int = 1,
    drop_rate: float = 0.4,
    eta_min: float = 1e-6,
    fast_dev_run: bool = False,
    first_out_features: int = 1024,
    in_features: int = IN_FEATURES,
    learning_rate: float = 3e-4,
    loss: str = "CELoss",
    max_epochs: int = 200,
    model_name: str = "linear",
    num_blocks: int = 2,
    num_classes: int = 250,
    num_workers: int = 2,
    overfit_batches: int = 0,
    optimizer: str = "AdamW",
    patience: int = 20,
    precision: int = 16,
    project: str = "asl-sign-detection-kaggle",
    scheduler: str = "CosineAnnealingLR",
    swa: bool = False,
    val_fold: float = 2.0,
    weight_decay: float = 1e-6,
):
    pl.seed_everything(SEED, workers=True)

    if fast_dev_run:
        num_workers = 0

    data_module = ASLDataModule(
        batch_size=batch_size,
        data_csv_path=TRAIN_CSV_PATH,
        in_features=in_features,
        num_workers=num_workers,
        val_fold=val_fold,
    )

    module = ASLModule(
        drop_rate=drop_rate,
        eta_min=eta_min,
        first_out_features=first_out_features,
        in_features=in_features,
        learning_rate=learning_rate,
        loss=loss,
        max_epochs=max_epochs,
        model_name=model_name,
        num_blocks=num_blocks,
        num_classes=num_classes,
        optimizer=optimizer,
        scheduler=scheduler,
        weight_decay=weight_decay,
    )

    logger, callbacks = load_logger_and_callbacks(
        fast_dev_run=fast_dev_run,
        metrics={"val_loss": "min", "val_acc": "max", "val_f1": "max"},
        overfit_batches=overfit_batches,
        patience=patience,
        project=project,
        val_fold=val_fold,
    )

    trainer = pl.Trainer(
        accelerator=accelerator,
        benchmark=True,
        devices=devices,
        callbacks=callbacks,
        fast_dev_run=fast_dev_run,
        logger=logger,
        log_every_n_steps=5,
        max_epochs=max_epochs,
        num_sanity_val_steps=0,
        overfit_batches=overfit_batches,
        precision=precision,
        strategy="ddp" if devices > 1 else None,
    )

    trainer.fit(module, datamodule=data_module)

    return module

module = train()