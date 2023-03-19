import os
import json
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

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

        x1m = torch.mean(face_x, 0).reshape(468,3)
        x2m = torch.mean(lefth_x, 0).reshape(21,3)
        x3m = torch.mean(pose_x, 0).reshape(33,3)
        x4m = torch.mean(righth_x, 0).reshape(21,3)

        x1s = torch.std(face_x, 0).reshape(468,3)
        x2s = torch.std(lefth_x, 0).reshape(21,3)
        x3s = torch.std(pose_x, 0).reshape(33,3)
        x4s = torch.std(righth_x, 0).reshape(21,3)

        xfeat = torch.cat([x1m,x1s, x2m,x2s, x3m,x3s, x4m,x4s], axis=0)
        xfeat = torch.where(torch.isnan(xfeat), torch.tensor(0.0, dtype=torch.float32), xfeat)

        return xfeat
def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)
def convert_row(row,feature_converter):
     x = load_relevant_data_subset(os.path.join("./asl-signs", row[1].path))
     x = feature_converter(torch.tensor(x)).cpu().numpy()
     return x, row[1].label
def convert_and_save_data(feature_converter):
    df = pd.read_csv(TRAIN_FILE)
    df['label'] = df['sign'].map(label_map)
    total = df.shape[0]
    if QUICK_TEST:
        total = QUICK_LIMIT
    npdata = np.zeros((df.shape[0], 1086,3))
    nplabels = np.zeros(total)
    for i, row in tqdm(enumerate(df.iterrows()), total=total):
        (x, y) = convert_row(row,feature_converter)
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


feature_converter = FeatureGen()

convert_and_save_data(feature_converter)