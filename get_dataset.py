from sklearn.model_selection import StratifiedGroupKFold
import tensorflow as tf
import numpy as np
from types import SimpleNamespace
from Model_Load import load_relevant_data_subset
import pandas as pd
import tqdm
TRAIN_CSV_PATH ='./asl-signs/train.csv'
train = pd.read_csv(TRAIN_CSV_PATH)
N_SAMPLES = len(train)

cfg = SimpleNamespace()
cfg.PREPROCESS_DATA = False
cfg.TRAIN_MODEL = False
cfg.N_ROWS = 543
cfg.N_DIMS = 3
cfg.DIM_NAMES = ['x', 'y', 'z']
cfg.SEED = 42
cfg.NUM_CLASSES = 250
cfg.IS_INTERACTIVE = True
cfg.VERBOSE = 2
cfg.INPUT_SIZE = 32
cfg.BATCH_ALL_SIGNS_N = 4
cfg.BATCH_SIZE = 256
cfg.N_EPOCHS = 50
cfg.LR_MAX = 1e-3
cfg.N_WARMUP_EPOCHS = 0
cfg.WD_RATIO = 0.05
cfg.MASK_VAL = 4237

LIPS_IDXS0 = np.array([
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
])
LEFT_HAND_IDXS0 = np.arange(468, 489)
RIGHT_HAND_IDXS0 = np.arange(522, 543)
POSE_IDXS0 = np.arange(502, 512)
LANDMARK_IDXS0 = np.concatenate((LIPS_IDXS0, LEFT_HAND_IDXS0, RIGHT_HAND_IDXS0, POSE_IDXS0))
HAND_IDXS0 = np.concatenate((LEFT_HAND_IDXS0, RIGHT_HAND_IDXS0), axis=0)
N_COLS = LANDMARK_IDXS0.size
LIPS_IDXS = np.argwhere(np.isin(LANDMARK_IDXS0, LIPS_IDXS0)).squeeze()
LEFT_HAND_IDXS = np.argwhere(np.isin(LANDMARK_IDXS0, LEFT_HAND_IDXS0)).squeeze()
RIGHT_HAND_IDXS = np.argwhere(np.isin(LANDMARK_IDXS0, RIGHT_HAND_IDXS0)).squeeze()
HAND_IDXS = np.argwhere(np.isin(LANDMARK_IDXS0, HAND_IDXS0)).squeeze()
POSE_IDXS = np.argwhere(np.isin(LANDMARK_IDXS0, POSE_IDXS0)).squeeze()


class PreprocessLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(PreprocessLayer, self).__init__()

    def pad_edge(self, t, repeats, side):
        if side == 'LEFT':
            return tf.concat((tf.repeat(t[:1], repeats=repeats, axis=0), t), axis=0)
        elif side == 'RIGHT':
            return tf.concat((t, tf.repeat(t[-1:], repeats=repeats, axis=0)), axis=0)

    @tf.function(
        input_signature=(tf.TensorSpec(shape=[None, cfg.N_ROWS, cfg.N_DIMS], dtype=tf.float32),),
    )
    def call(self, data0):
        # Number of Frames in Video
        N_FRAMES0 = tf.shape(data0)[0]

        # Keep only non-empty frames in data
        frames_hands_nansum = tf.experimental.numpy.nanmean(tf.gather(data0, HAND_IDXS0, axis=1), axis=[1, 2])
        non_empty_frames_idxs = tf.where(frames_hands_nansum > 0)
        non_empty_frames_idxs = tf.squeeze(non_empty_frames_idxs, axis=1)
        data = tf.gather(data0, non_empty_frames_idxs, axis=0)

        non_empty_frames_idxs = tf.cast(non_empty_frames_idxs, tf.float32)

        # Number of non-empty frames
        N_FRAMES = tf.shape(data)[0]
        data = tf.gather(data, LANDMARK_IDXS0, axis=1)

        if N_FRAMES < cfg.INPUT_SIZE:
            # Video fits in cfg.INPUT_SIZE
            non_empty_frames_idxs = tf.pad(non_empty_frames_idxs, [[0, cfg.INPUT_SIZE - N_FRAMES]], constant_values=-1)
            data = tf.pad(data, [[0, cfg.INPUT_SIZE - N_FRAMES], [0, 0], [0, 0]], constant_values=0)
            data = tf.where(tf.math.is_nan(data), 0.0, data)
            return data, non_empty_frames_idxs
        else:
            # Video needs to be downsampled to cfg.INPUT_SIZE
            if N_FRAMES < cfg.INPUT_SIZE ** 2:
                repeats = tf.math.floordiv(cfg.INPUT_SIZE * cfg.INPUT_SIZE, N_FRAMES0)
                data = tf.repeat(data, repeats=repeats, axis=0)
                non_empty_frames_idxs = tf.repeat(non_empty_frames_idxs, repeats=repeats, axis=0)

            # Pad To Multiple Of Input Size
            pool_size = tf.math.floordiv(len(data), cfg.INPUT_SIZE)
            if tf.math.mod(len(data), cfg.INPUT_SIZE) > 0:
                pool_size += 1
            if pool_size == 1:
                pad_size = (pool_size * cfg.INPUT_SIZE) - len(data)
            else:
                pad_size = (pool_size * cfg.INPUT_SIZE) % len(data)

            # Pad Start/End with Start/End value
            pad_left = tf.math.floordiv(pad_size, 2) + tf.math.floordiv(cfg.INPUT_SIZE, 2)
            pad_right = tf.math.floordiv(pad_size, 2) + tf.math.floordiv(cfg.INPUT_SIZE, 2)
            if tf.math.mod(pad_size, 2) > 0:
                pad_right += 1

            # Pad By Concatenating Left/Right Edge Values
            data = self.pad_edge(data, pad_left, 'LEFT')
            data = self.pad_edge(data, pad_right, 'RIGHT')

            # Pad Non Empty Frame Indices
            non_empty_frames_idxs = self.pad_edge(non_empty_frames_idxs, pad_left, 'LEFT')
            non_empty_frames_idxs = self.pad_edge(non_empty_frames_idxs, pad_right, 'RIGHT')

            # Reshape to Mean Pool
            data = tf.reshape(data, [cfg.INPUT_SIZE, -1, N_COLS, cfg.N_DIMS])
            non_empty_frames_idxs = tf.reshape(non_empty_frames_idxs, [cfg.INPUT_SIZE, -1])

            # Mean Pool
            data = tf.experimental.numpy.nanmean(data, axis=1)
            non_empty_frames_idxs = tf.experimental.numpy.nanmean(non_empty_frames_idxs, axis=1)
            data = tf.where(tf.math.is_nan(data), 0.0, data)
            return data, non_empty_frames_idxs


preprocess_layer = PreprocessLayer()
def get_data(file_path):
    data = load_relevant_data_subset(file_path)
    data = preprocess_layer(data)
    return data
def get_x_y():
    # Create arrays to save data
    X = np.zeros([N_SAMPLES, cfg.INPUT_SIZE, N_COLS, cfg.N_DIMS], dtype=np.float32)
    y = np.zeros([N_SAMPLES], dtype=np.int32)
    NON_EMPTY_FRAME_IDXS = np.full([N_SAMPLES, cfg.INPUT_SIZE], -1, dtype=np.float32)

    for row_idx, (file_path, sign_ord) in enumerate(tqdm(train[['file_path', 'sign_ord']].values)):
        if row_idx % 5000 == 0:
            print(f'Generated {row_idx}/{N_SAMPLES}')

        data, non_empty_frame_idxs = get_data(file_path)
        X[row_idx] = data
        y[row_idx] = sign_ord
        NON_EMPTY_FRAME_IDXS[row_idx] = non_empty_frame_idxs
        if np.isnan(data).sum() > 0: return data

if cfg.PREPROCESS_DATA:
    X, y, NON_EMPTY_FRAME_IDXS = get_x_y()
else:
    X = np.load('/kaggle/input/gislr-dataset-public/X.npy')
    y = np.load('/kaggle/input/gislr-dataset-public/y.npy')
    NON_EMPTY_FRAME_IDXS = np.load('/kaggle/input/gislr-dataset-public/NON_EMPTY_FRAME_IDXS.npy')

LIPS_MEAN_X = np.zeros([LIPS_IDXS.size], dtype=np.float32)
LIPS_MEAN_Y = np.zeros([LIPS_IDXS.size], dtype=np.float32)
LIPS_STD_X = np.zeros([LIPS_IDXS.size], dtype=np.float32)
LIPS_STD_Y = np.zeros([LIPS_IDXS.size], dtype=np.float32)

for col, ll in enumerate(
        tqdm(np.transpose(X[:, :, LIPS_IDXS], [2, 3, 0, 1]).reshape([LIPS_IDXS.size, cfg.N_DIMS, -1]))):
    for dim, l in enumerate(ll):
        v = l[np.nonzero(l)]
        if dim == 0:  # X
            LIPS_MEAN_X[col] = v.mean()
            LIPS_STD_X[col] = v.std()
        if dim == 1:  # Y
            LIPS_MEAN_Y[col] = v.mean()
            LIPS_STD_Y[col] = v.std()

LIPS_MEAN = np.array([LIPS_MEAN_X, LIPS_MEAN_Y]).T
LIPS_STD = np.array([LIPS_STD_X, LIPS_STD_Y]).T

LEFT_HANDS_MEAN_X = np.zeros([LEFT_HAND_IDXS.size], dtype=np.float32)
LEFT_HANDS_MEAN_Y = np.zeros([LEFT_HAND_IDXS.size], dtype=np.float32)
LEFT_HANDS_STD_X = np.zeros([LEFT_HAND_IDXS.size], dtype=np.float32)
LEFT_HANDS_STD_Y = np.zeros([LEFT_HAND_IDXS.size], dtype=np.float32)
# RIGHT HAND
RIGHT_HANDS_MEAN_X = np.zeros([RIGHT_HAND_IDXS.size], dtype=np.float32)
RIGHT_HANDS_MEAN_Y = np.zeros([RIGHT_HAND_IDXS.size], dtype=np.float32)
RIGHT_HANDS_STD_X = np.zeros([RIGHT_HAND_IDXS.size], dtype=np.float32)
RIGHT_HANDS_STD_Y = np.zeros([RIGHT_HAND_IDXS.size], dtype=np.float32)

for col, ll in enumerate(
        tqdm(np.transpose(X[:, :, HAND_IDXS], [2, 3, 0, 1]).reshape([HAND_IDXS.size, cfg.N_DIMS, -1]))):
    for dim, l in enumerate(ll):
        v = l[np.nonzero(l)]
        if dim == 0:  # X
            if col < RIGHT_HAND_IDXS.size:  # LEFT HAND
                LEFT_HANDS_MEAN_X[col] = v.mean()
                LEFT_HANDS_STD_X[col] = v.std()
            else:
                RIGHT_HANDS_MEAN_X[col - LEFT_HAND_IDXS.size] = v.mean()
                RIGHT_HANDS_STD_X[col - LEFT_HAND_IDXS.size] = v.std()
        if dim == 1:  # Y
            if col < RIGHT_HAND_IDXS.size:  # LEFT HAND
                LEFT_HANDS_MEAN_Y[col] = v.mean()
                LEFT_HANDS_STD_Y[col] = v.std()
            else:  # RIGHT HAND
                RIGHT_HANDS_MEAN_Y[col - LEFT_HAND_IDXS.size] = v.mean()
                RIGHT_HANDS_STD_Y[col - LEFT_HAND_IDXS.size] = v.std()

LEFT_HANDS_MEAN = np.array([LEFT_HANDS_MEAN_X, LEFT_HANDS_MEAN_Y]).T
LEFT_HANDS_STD = np.array([LEFT_HANDS_STD_X, LEFT_HANDS_STD_Y]).T
RIGHT_HANDS_MEAN = np.array([RIGHT_HANDS_MEAN_X, RIGHT_HANDS_MEAN_Y]).T
RIGHT_HANDS_STD = np.array([RIGHT_HANDS_STD_X, RIGHT_HANDS_STD_Y]).T

# POSE
POSE_MEAN_X = np.zeros([POSE_IDXS.size], dtype=np.float32)
POSE_MEAN_Y = np.zeros([POSE_IDXS.size], dtype=np.float32)
POSE_STD_X = np.zeros([POSE_IDXS.size], dtype=np.float32)
POSE_STD_Y = np.zeros([POSE_IDXS.size], dtype=np.float32)

for col, ll in enumerate(
        tqdm(np.transpose(X[:, :, POSE_IDXS], [2, 3, 0, 1]).reshape([POSE_IDXS.size, cfg.N_DIMS, -1]))):
    for dim, l in enumerate(ll):
        v = l[np.nonzero(l)]
        if dim == 0:  # X
            POSE_MEAN_X[col] = v.mean()
            POSE_STD_X[col] = v.std()
        if dim == 1:  # Y
            POSE_MEAN_Y[col] = v.mean()
            POSE_STD_Y[col] = v.std()

POSE_MEAN = np.array([POSE_MEAN_X, POSE_MEAN_Y]).T
POSE_STD = np.array([POSE_STD_X, POSE_STD_Y]).T


def get_train_batch_all_signs(X, y, NON_EMPTY_FRAME_IDXS, n=cfg.BATCH_ALL_SIGNS_N):
    # Arrays to store batch in
    X_batch = np.zeros([cfg.NUM_CLASSES * n, cfg.INPUT_SIZE, N_COLS, cfg.N_DIMS], dtype=np.float32)
    y_batch = np.arange(0, cfg.NUM_CLASSES, step=1 / n, dtype=np.float32).astype(np.int64)
    non_empty_frame_idxs_batch = np.zeros([cfg.NUM_CLASSES * n, cfg.INPUT_SIZE], dtype=np.float32)

    # Dictionary mapping ordinally encoded sign to corresponding sample indices
    CLASS2IDXS = {}
    for i in range(cfg.NUM_CLASSES):
        CLASS2IDXS[i] = np.argwhere(y == i).squeeze().astype(np.int32)

    while True:
        # Fill batch arrays
        for i in range(cfg.NUM_CLASSES):
            idxs = np.random.choice(CLASS2IDXS[i], n)
            X_batch[i * n:(i + 1) * n] = X[idxs]
            non_empty_frame_idxs_batch[i * n:(i + 1) * n] = NON_EMPTY_FRAME_IDXS[idxs]

        yield {'frames': X_batch, 'non_empty_frame_idxs': non_empty_frame_idxs_batch}, y_batch

