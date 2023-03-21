import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import warnings
from tqdm import tqdm
import pandas as pd
import os
from sklearn.metrics import f1_score
import math
ROWS_PER_FRAME = 543


class FeatureGen(nn.Module):
    def __init__(self):
        super(FeatureGen, self).__init__()
        pass

    def forward(self, x):
        face_x = x[:, :468, :].contiguous().view(-1, 468 * 3)
        lefth_x = x[:, 468:489, :].contiguous().view(-1, 21 * 3)
        pose_x = x[:, 489:522, :].contiguous().view(-1, 33 * 3)
        righth_x = x[:, 522:, :].contiguous().view(-1, 21 * 3)

        lefth_x = lefth_x[~torch.any(torch.isnan(lefth_x), dim=1), :]
        righth_x = righth_x[~torch.any(torch.isnan(righth_x), dim=1), :]

        x1m = torch.mean(face_x, 0).reshape(468, 3)
        x2m = torch.mean(lefth_x, 0).reshape(21, 3)
        x3m = torch.mean(pose_x, 0).reshape(33, 3)
        x4m = torch.mean(righth_x, 0).reshape(21, 3)

        x1s = torch.std(face_x, 0).reshape(468, 3)
        x2s = torch.std(lefth_x, 0).reshape(21, 3)
        x3s = torch.std(pose_x, 0).reshape(33, 3)
        x4s = torch.std(righth_x, 0).reshape(21, 3)

        xfeat = torch.cat([x1m, x1s, x2m, x2s, x3m, x3s, x4m, x4s], axis=0)
        xfeat = torch.where(torch.isnan(xfeat), torch.tensor(0.0, dtype=torch.float32), xfeat)

        return xfeat


def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)


def convert_row(row, feature_converter):
    x = load_relevant_data_subset(os.path.join("./asl-signs", row[1].path))
    x = feature_converter(torch.tensor(x)).cpu().numpy()
    return x, row[1].label

class ArcMarginProduct(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        scale=30.0,
        margin=0.50,
        easy_margin=False,
        ls_eps=0.0,
    ):
        super(ArcMarginProduct, self).__init__()
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

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        return output

def convert_and_save_data(feature_converter):
    df = pd.read_csv(TRAIN_FILE)
    df['label'] = df['sign'].map(label_map)
    total = df.shape[0]
    if QUICK_TEST:
        total = QUICK_LIMIT
    npdata = np.zeros((df.shape[0], 1086, 3))
    nplabels = np.zeros(total)
    for i, row in tqdm(enumerate(df.iterrows()), total=total):
        (x, y) = convert_row(row, feature_converter)
        npdata[i, :] = x
        nplabels[i] = y
        if QUICK_TEST and i == QUICK_LIMIT - 1:
            break

    np.save("feature_data.npy", npdata)
    np.save("feature_labels.npy", nplabels)


QUICK_TEST = False
QUICK_LIMIT = 200

LANDMARK_FILES_DIR = "./asl-signs/train_landmark_files"
TRAIN_FILE = "./asl-signs/train.csv"
label_map = json.load(open("./asl-signs/sign_to_prediction_index_map.json", "r"))
class CNN_Model(nn.Module):
    def __init__(self,dependent = False):
        super(CNN_Model, self).__init__()
        self.CNN = nn.Sequential(
            nn.Conv1d(3, 4, kernel_size=3),
            nn.GELU(),
            nn.BatchNorm1d(4),
            nn.Conv1d(4, 4, kernel_size=3, stride=2),
            nn.GELU(),
            nn.BatchNorm1d(4),
            nn.Dropout(0.4,inplace=False),
            nn.Conv1d(4, 6, kernel_size=3),
            nn.GELU(),
            nn.BatchNorm1d(6),
            nn.Conv1d(6, 6, kernel_size=3, stride=2),
            nn.GELU(),
            nn.BatchNorm1d(6),
            nn.Dropout(0.3,inplace=False))
        self.classifier1 = nn.Sequential(nn.Linear(1614, 807), nn.GELU(),
                                         nn.BatchNorm1d(807),
                                         nn.Dropout(0.2,inplace=False),
                                         nn.Linear(807, 250))

        self.dependent = dependent
    def forward(self, x):
        if not self.dependent:
            x = x.permute(0, 2, 1)

        X1 = self.CNN(x)
        return self.classifier1(X1.view(-1, 1614))

class RGB_Linear_Model(nn.Module):
    def __init__(self):
        super(RGB_Linear_Model, self).__init__()
        self.RGB_Linear = nn.Sequential(nn.Dropout(0.2), nn.Linear(3508, 877),nn.BatchNorm1d(877),
                                        nn.Dropout(0.2,inplace=False), nn.GELU(),
                                        nn.Linear(877, 250),nn.BatchNorm1d(250))

    def forward(self, x):
        return self.RGB_Linear(x)


class Multiple_Ensemble(nn.Module):
    def __init__(self):
        super(Multiple_Ensemble, self).__init__()
        self.CNN = CNN_Model(dependent=True)
        self.boost_rate = nn.Parameter(torch.tensor(0.5, requires_grad=True, device="cuda"))
        self.classifier2 = RGB_Linear_Model()
        self.classifier3_encoder = nn.Sequential(
            nn.Conv1d(3, 4, kernel_size=4),
            nn.GELU(),
            nn.BatchNorm1d(4),
            nn.AdaptiveAvgPool1d(400))
        self.classifier3 =  nn.GRUCell(1600,250)


        self.boost_rate_1 = nn.Parameter(torch.tensor(0.33, requires_grad=True, device="cuda"))
        self.boost_rate_2 = nn.Parameter(torch.tensor(0.33, requires_grad=True, device="cuda"))
        self.boost_rate_3 = nn.Parameter(torch.tensor(0.33, requires_grad=True, device="cuda"))
    def forward(self, x):
        x = x.permute(0, 2, 1)
        X1 = self.CNN(x)

        X2 = torch.cat([x.reshape(-1, 3258), X1], dim=1)
        X2 = self.classifier2(X2)

        X3 = self.classifier3_encoder(x)
        X3 =  self.classifier3(X3.reshape(-1,1600),X2)

        return self.boost_rate_1*X1 +self.boost_rate_2*X2 +self.boost_rate_3*X3


import onnx
from onnx_tf.backend import prepare


class ASLData(Dataset):
    def __init__(self, datax, datay):
        self.datax = datax
        self.datay = datay

    def __getitem__(self, index):
        return self.datax[index, :], self.datay[index]

    def __len__(self):
        return len(self.datay)


warnings.filterwarnings(action='ignore')
if __name__ == '__main__':

    best_score = 0
    LANDMARK_FILES_DIR = "./asl-signs/train_landmark_files"
    TRAIN_FILE = "./asl-signs/train.csv"
    label_map = json.load(open("./asl-signs/sign_to_prediction_index_map.json", "r"))

    datax = np.load("./feature_data.npy")
    datay = np.load("./feature_labels.npy")
    EPOCHS =300
    BATCH_SIZE = 512


    model = Multiple_Ensemble().cuda()
   # saved_cnn = torch.load('./saved_CNN.pt')
   # model.CNN.load_state_dict(saved_cnn)
    #saved_model = torch.load('./best_model.pt')
    #model.load_state_dict(saved_model)

    opt = torch.optim.AdamW(model.parameters(), lr=0.02,weight_decay=1e-6)
    criterion = nn.CrossEntropyLoss()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=20, eta_min=1e-5)

    for i in range(EPOCHS):
        model.train()
        if i%10 == 0:
            trainx, testx, trainy, testy = train_test_split(datax, datay, test_size=0.35, random_state=random.randint(1,100))

            train_data = ASLData(trainx, trainy)
            valid_data = ASLData(testx, testy)

            train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, num_workers=4, shuffle=True)
            val_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)

        train_loss_sum = 0.
        train_correct = 0
        train_total = 0
        train_bar = train_loader
        for x, y in tqdm(iter(train_bar)):
            x = torch.Tensor(x).float().cuda()
            y = torch.Tensor(y).long().cuda()
            y_pred = model(x)

            loss = criterion(y_pred, y)

            loss.backward()
            opt.step()
            opt.zero_grad()

            train_loss_sum += loss.item()
            train_correct += np.sum((np.argmax(y_pred.detach().cpu().numpy(), axis=1) == y.cpu().numpy()))
            train_total += 1
            sched.step()

        val_loss_sum = 0.
        val_correct = 0
        val_total = 0
        model.eval()
        preds, trues = [], []
        for x, y in tqdm(iter(val_loader)):
            x = torch.Tensor(x).float().cuda()
            y = torch.Tensor(y).long().cuda()

            with torch.no_grad():
                y_pred= model(x)
                loss = criterion(y_pred, y)
                val_loss_sum += loss.item()
                val_correct += np.sum((np.argmax(y_pred.cpu().numpy(), axis=1) == y.cpu().numpy()))
                val_total += 1
                preds += y_pred.argmax(1).detach().cpu().numpy().tolist()
                trues += y.detach().cpu().numpy().tolist()
        _val_score = f1_score(trues, preds, average='macro')
        print(
            f"Epoch:{i} > Train Loss: {(train_loss_sum / train_total):.04f}, Train Acc: {train_correct / len(train_data):0.04f}")
        print(
            f"Epoch:{i} > Val Loss: {(val_loss_sum / val_total):.04f}, Val Acc: {val_correct / len(valid_data):0.04f}, Val F1: {_val_score:0.04f}")
        print("=" * 50)
        torch.save(model.state_dict(), './saved_model.pt')
        if _val_score > best_score:
            torch.save(model.state_dict(), './best_model.pt')
            best_score = _val_score
