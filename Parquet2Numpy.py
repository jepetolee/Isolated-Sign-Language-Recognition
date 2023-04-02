import os
import json
from tqdm import tqdm
import numpy as np
import pandas as pd

import tensorflow as tf
ROWS_PER_FRAME = 543

import tensorflow as tf
def tf_nan_mean(x, axis=0):
    return tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), x), axis=axis) / tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), tf.ones_like(x)), axis=axis)

def tf_nan_std(x, axis=0):
    d = x - tf_nan_mean(x, axis=axis)
    return tf.math.sqrt(tf_nan_mean(d * d, axis=axis))
class FeatureGen(tf.keras.layers.Layer):
    def __init__(self):
        super(FeatureGen, self).__init__()
    def call(self, x):
        face_x = x[:, :468, :]
        lefth_x = x[:, 468:489, :]
        pose_x = x[:, 489:522, :]
        righth_x = x[:, 522:, :]

        lefth_x =tf.where(tf.math.is_nan(lefth_x), tf.zeros_like(lefth_x), lefth_x)
        righth_x =tf.where(tf.math.is_nan(righth_x), tf.zeros_like(righth_x), righth_x)


        x1m = tf_nan_mean(face_x, 0)
        x2m = tf_nan_mean(lefth_x, 0)
        x3m =tf_nan_mean(pose_x, 0)
        x4m =tf_nan_mean(righth_x, 0)
        x1s = tf_nan_std(face_x, 0)
        x2s = tf_nan_std(lefth_x, 0)
        x3s = tf_nan_std(pose_x, 0)
        x4s = tf_nan_std(righth_x, 0)

        xfeat = tf.concat([x1m, x1s, x2m, x2s, x3m, x3s, x4m, x4s], axis=0)
        xfeat = tf.where(tf.math.is_nan(xfeat), tf.zeros_like(xfeat), xfeat)
        xfeat = tf.reshape(xfeat,[-1,1086,3])
        return xfeat
def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)

def convert_row(row,feature_converter):
     x = load_relevant_data_subset(os.path.join("./asl-signs", row[1].path))
     x = feature_converter(tf.convert_to_tensor(x)).numpy()
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


#feature_converter = FeatureGen()

#convert_and_save_data(feature_converter)