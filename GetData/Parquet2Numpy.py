import os
import json
from tqdm import tqdm
import numpy as np
import pandas as pd

import tensorflow as tf
ROWS_PER_FRAME = 543


DROP_Z = False

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


INPUT_SHAPE = (NUM_FRAMES,LANDMARKS,3)

FLAT_INPUT_SHAPE = (INPUT_SHAPE[0] + 2 * (SEGMENTS + 1)) * INPUT_SHAPE[1]

def tf_nan_mean(x, axis=0):
    return tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), x), axis=axis) / tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), tf.ones_like(x)), axis=axis)

def tf_nan_std(x, axis=0):
    d = x - tf_nan_mean(x, axis=axis)
    return tf.math.sqrt(tf_nan_mean(d * d, axis=axis))

def means_and_stds(x, axis=0):
    # Get means and stds
    x_mean = tf_nan_mean(x, axis=0)
    x_std  = tf_nan_std(x,  axis=0)

    x_out = tf.concat([x_mean, x_std], axis=0)
    x_out = tf.where(tf.math.is_finite(x_out), x_out, tf.zeros_like(x_out))
    return x_out

class FeatureGen(tf.keras.layers.Layer):
    def __init__(self):
        super(FeatureGen, self).__init__()

    def call(self, x_in):
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
        x_list = [means_and_stds(_x, axis=0) for _x in x_list]

        x_list.append(means_and_stds(x, axis=0))

        ## Resize only dimension 0. Resize can't handle nan, so replace nan with that dimension's avg value to reduce impact.
        x = tf.image.resize(tf.where(tf.math.is_finite(x), x, tf_nan_mean(x, axis=0)), [NUM_FRAMES, LANDMARKS])
        x = tf.reshape(x, (INPUT_SHAPE[0] * INPUT_SHAPE[1],INPUT_SHAPE[2]))
        x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)
        x_list.append(x)
        x = tf.concat(x_list, axis=0)
        return x


def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)

def convert_row(row,feature_converter):
     x = load_relevant_data_subset(os.path.join("../asl-signs", row[1].path))
     x = feature_converter(tf.convert_to_tensor(x)).numpy()
     return x, row[1].label


def convert_and_save_data():
    df = pd.read_csv(TRAIN_FILE)
    df['label'] = df['sign'].map(label_map)
    total = df.shape[0]
    if QUICK_TEST:
        total = QUICK_LIMIT
    npdata = np.zeros((total, 1932,3))
    nplabels = np.zeros(total)
    feature_converter = FeatureGen()
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




#convert_and_save_data()