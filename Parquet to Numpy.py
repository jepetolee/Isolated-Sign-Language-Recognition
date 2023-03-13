import os

import json
from tqdm import tqdm
import numpy as np
import pandas as pd

import tensorflow as tf

QUICK_TEST = False
QUICK_LIMIT = 200

LANDMARK_FILES_DIR = "/kaggle/input/asl-signs/train_landmark_files"
TRAIN_FILE = "/kaggle/input/asl-signs/train.csv"
label_map = json.load(open("/kaggle/input/asl-signs/sign_to_prediction_index_map.json", "r"))


ROWS_PER_FRAME = 543
def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)

def right_hand_percentage(x):
    right = tf.gather(x, right_hand_landmarks, axis=1)
    left = tf.gather(x, left_hand_landmarks, axis=1)
    right_count = tf.reduce_sum(tf.where(tf.math.is_nan(right), tf.zeros_like(right), tf.ones_like(right)))
    left_count = tf.reduce_sum(tf.where(tf.math.is_nan(left), tf.zeros_like(left), tf.ones_like(left)))
    return right_count / (left_count+right_count)

NUM_FRAMES = 15
SEGMENTS = 3

LEFT_HAND_OFFSET = 468
POSE_OFFSET = LEFT_HAND_OFFSET+21
RIGHT_HAND_OFFSET = POSE_OFFSET+33

## average over the entire face, and the entire 'pose'
averaging_sets = [[0, 468], [POSE_OFFSET, 33]]

lip_landmarks = [61, 185, 40, 39, 37,  0, 267, 269, 270, 409,
                 291,146, 91,181, 84, 17, 314, 405, 321, 375,
                 78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
                 95, 88, 178, 87, 14,317, 402, 318, 324, 308]
left_hand_landmarks = list(range(LEFT_HAND_OFFSET, LEFT_HAND_OFFSET+21))
right_hand_landmarks = list(range(RIGHT_HAND_OFFSET, RIGHT_HAND_OFFSET+21))

point_landmarks = [item for sublist in [lip_landmarks, left_hand_landmarks, right_hand_landmarks] for item in sublist]

LANDMARKS = len(point_landmarks) + len(averaging_sets)
print(LANDMARKS)
INPUT_SHAPE = (NUM_FRAMES,LANDMARKS*3)
def tf_nan_mean(x, axis=0):
    return tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), x), axis=axis) / tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), tf.ones_like(x)), axis=axis)

def tf_nan_std(x, axis=0):
    d = x - tf_nan_mean(x, axis=axis)
    return tf.math.sqrt(tf_nan_mean(d * d, axis=axis))

def flatten_means_and_stds(x, axis=0):
    # Get means and stds
    x_mean = tf_nan_mean(x, axis=0)
    x_std  = tf_nan_std(x,  axis=0)

    x_out = tf.concat([x_mean, x_std], axis=0)
    x_out = tf.reshape(x_out, (1, INPUT_SHAPE[1]*2))
    x_out = tf.where(tf.math.is_finite(x_out), x_out, tf.zeros_like(x_out))
    return x_out


class FeatureGen(tf.keras.layers.Layer):
    def __init__(self):
        super(FeatureGen, self).__init__()

    def call(self, x_in):
        #         print(right_hand_percentage(x))
        x_list = [tf.expand_dims(tf_nan_mean(x_in[:, av_set[0]:av_set[0] + av_set[1], :], axis=1), axis=1) for av_set in
                  averaging_sets]
        x_list.append(tf.gather(x_in, point_landmarks, axis=1))
        x = tf.concat(x_list, 1)

        x_padded = x
        for i in range(SEGMENTS):
            p0 = tf.where(((tf.shape(x_padded)[0] % SEGMENTS) > 0) & ((i % 2) != 0), 1, 0)
            p1 = tf.where(((tf.shape(x_padded)[0] % SEGMENTS) > 0) & ((i % 2) == 0), 1, 0)
            paddings = [[p0, p1], [0, 0], [0, 0]]
            x_padded = tf.pad(x_padded, paddings, mode="SYMMETRIC")
        x_list = tf.split(x_padded, SEGMENTS)
        x_list = [flatten_means_and_stds(_x, axis=0) for _x in x_list]

        x_list.append(flatten_means_and_stds(x, axis=0))

        ## Resize only dimension 0. Resize can't handle nan, so replace nan with that dimension's avg value to reduce impact.
        x = tf.image.resize(tf.where(tf.math.is_finite(x), x, tf_nan_mean(x, axis=0)), [NUM_FRAMES, LANDMARKS])
        x = tf.reshape(x, (1, INPUT_SHAPE[0] * INPUT_SHAPE[1]))
        x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)
        x_list.append(x)
        x = tf.concat(x_list, axis=1)
        return x


feature_converter = FeatureGen()

## One tests symbolic tensor, the other tests real data.
print(feature_converter(tf.keras.Input((543, 3), dtype=tf.float32, name="inputs")))
feature_converter(load_relevant_data_subset(f'/kaggle/input/asl-signs/{pd.read_csv(TRAIN_FILE).path[1]}'))


def convert_row(row, right_handed=True):
    x = load_relevant_data_subset(os.path.join("/kaggle/input/asl-signs", row[1].path))
    x = feature_converter(tf.convert_to_tensor(x)).cpu().numpy()
    return x, row[1].label


right_handed_signer = [26734, 28656, 25571, 62590, 29302,
                       49445, 53618, 18796, 4718, 2044,
                       37779, 30680]
left_handed_signer = [16069, 32319, 36257, 22343, 27610,
                      61333, 34503, 55372, ]
both_hands_signer = [37055, ]

messy = [29302, ]


def convert_and_save_data():
    df = pd.read_csv(TRAIN_FILE)
    df['label'] = df['sign'].map(label_map)
    total = df.shape[0]
    if QUICK_TEST:
        total = QUICK_LIMIT
    npdata = np.zeros((total, INPUT_SHAPE[0] * INPUT_SHAPE[1] + (SEGMENTS + 1) * INPUT_SHAPE[1] * 2))
    nplabels = np.zeros(total)
    for i, row in tqdm(enumerate(df.iterrows()), total=total):
        (x, y) = convert_row(row)
        npdata[i, :] = x
        nplabels[i] = y
        if QUICK_TEST and i == QUICK_LIMIT - 1:
            break

    np.save("feature_data.npy", npdata)
    np.save("feature_labels.npy", nplabels)
X = np.load("feature_data.npy")
y = np.load("feature_labels.npy")
print(X.shape, y.shape)

print(X[0, :].shape, X[0, :])