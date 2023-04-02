import random
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from sklearn.model_selection import train_test_split
import warnings


class Conv(tf.keras.layers.Layer):
    def __init__(self, num_outputs, dropout):
        super(Conv, self).__init__()

        self.conv1 = tf.keras.layers.Conv1D(num_outputs, kernel_size=2, strides=1,
                                            kernel_initializer='he_normal')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.act1 = tf.keras.layers.Activation('gelu')
        self.conv2 = tf.keras.layers.Conv1D(num_outputs, kernel_size=2, strides=2, activation='gelu',
                                            kernel_initializer='he_normal')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.act2 = tf.keras.layers.Activation('gelu')
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        return self.dropout(x)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'conv1': self.conv1,
            'bn1': self.bn1,
            'act1': self.act1,
            'conv2': self.conv2,
            'bn2': self.bn2,
            'act2': self.act2,
            'dropout': self.dropout
        })
        return config


class Identity_Block(tf.keras.layers.Layer):
    def __init__(self, num_outputs, dropout):
        super(Identity_Block, self).__init__()
        self.conv1 = tf.keras.layers.Conv1D(num_outputs, kernel_size=3, strides=1, padding='SAME',
                                            kernel_initializer='he_normal')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.act1 = tf.keras.layers.Activation('gelu')
        self.conv2 = tf.keras.layers.Conv1D(num_outputs, kernel_size=3, strides=2, padding='SAME',
                                            kernel_initializer='he_normal')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.act2 = tf.keras.layers.Activation('gelu')

        self.identity = tf.keras.layers.Conv1D(num_outputs, kernel_size=3, strides=2, padding='SAME',
                                               kernel_initializer='he_normal')
        self.identitybn = tf.keras.layers.BatchNormalization()
        self.add = tf.keras.layers.Add()
        self.act3 = tf.keras.layers.Activation('gelu')

    def call(self, inputs):
        shortcut = inputs
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        shortcut = self.identity(shortcut)
        shortcut = self.identitybn(shortcut)

        x = self.add([x, shortcut])
        x = self.act3(x)
        return self.dropout(x)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'conv1': self.conv1,
            'bn1': self.bn1,
            'act1': self.act1,
            'conv2': self.conv2,
            'bn2': self.bn2,
            'act2': self.act2,
            'identity': self.identity,
            'identitybn': self.identitybn,
            'add': self.add,
            'act3': self.act3,
            'dropout': self.dropout
        })
        return config

@tf.keras.utils.register_keras_serializable()
class Dense(tf.keras.layers.Layer):
    def __init__(self, num_outputs, dropout_exist=True, dropout=0.1):
        super(Dense, self).__init__()
        self.num_outputs = num_outputs
        self.dense = tf.keras.layers.Dense(self.num_outputs, activation='gelu', kernel_initializer='he_normal')
        self.bn = tf.keras.layers.BatchNormalization()
        self.dropout_exist = dropout_exist
        if self.dropout_exist:
            self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.bn(x)
        if self.dropout_exist:
            x = self.dropout(x)
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_outputs': self.num_outputs,
            'dense': self.dense,
            'bn': self.bn,
            'conv2': self.conv2,
            'bn2': self.bn2,
            'dropout_exist': self.dropout_exist
        })
        if self.dropout_exist:
            config.update({
                'dropout': self.dropout
            })
        return config

from Model_Load import *


class CNN_Model(tf.keras.Model):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.Conv1 = Conv(8, 0.2)
        self.Indentity1 = Identity_Block(8, 0.3)
        self.Conv2 = Conv(20, 0.2)
        self.Indentity2 = Identity_Block(20, 0.2)
        self.Conv3 = Conv(40, 0.1)
        self.Indentity3 = Identity_Block(40, 0.2)
        self.Conv4 = Conv(80, 0.2)
        self.Indentity4 = Identity_Block(80, 0.1)
        self.Flatten = tf.keras.layers.Flatten()
        self.Dense = Dense(250, dropout_exist=False)
        self.softmax = tf.keras.layers.Softmax(dtype="float32")

        self.m21Conv1 = Conv(6, 0.2)
        self.m21Indentity1 = Identity_Block(6, 0.2)
        self.m21Conv2 = Conv(12, 0.2)
        self.m21Indentity2 = Identity_Block(12, 0.2)
        self.m21Conv3 = Conv(24, 0.2)
        self.m21Indentity3 = Identity_Block(24, 0.2)
        self.m21Conv4 = Conv(48, 0.1)
        self.m21Dense = Dense(250, dropout_exist=False)

        self.m2Conv1 = Conv(5, 0.3)
        self.m2Indentity1 = Identity_Block(6, 0.2)
        self.m2Conv2 = Conv(10, 0.2)
        self.m2Indentity2 = Identity_Block(12, 0.2)
        self.m2Conv3 = Conv(15, 0.2)
        self.m2Indentity3 = Identity_Block(24, 0.2)
        self.m2Dense1 = Dense(320, dropout=0.2)
        self.m2Dense2 = Dense(250, dropout_exist=False)

        self.m3Dens1 = Dense(1500, dropout=0.2)
        self.m3Dens2 = Dense(500, dropout=0.1)
        self.m3Dens3 = Dense(250, dropout_exist=False)

        self.m7Conv1 = Conv(9, 0.2)
        self.m7Pool1 = tf.keras.layers.MaxPool1D(pool_size=2)
        self.m7Conv2 = Conv(27, 0.2)
        self.m7Pool2 = tf.keras.layers.MaxPool1D(pool_size=2)
        self.m7Conv3 = Conv(81, 0.2)
        self.m7Pool3 = tf.keras.layers.MaxPool1D(pool_size=2)
        self.m7Dense1 = Dense(320, dropout=0.1)
        self.m7Dense2 = Dense(250, dropout_exist=False)

        self.m8Conv1 = Conv(7, 0.2)
        self.m8Pool1 = tf.keras.layers.AvgPool1D(pool_size=1)
        self.m8Conv2 = Conv(21, 0.2)
        self.m8Pool2 = tf.keras.layers.AvgPool1D(pool_size=1)
        self.m8Conv3 = Conv(42, 0.2)
        self.m8Pool3 = tf.keras.layers.AvgPool1D(pool_size=1)
        self.m8Dense1 = Dense(320, dropout=0.2)
        self.m8Dense2 = Dense(250, dropout_exist=False)

        self.m4Dens1 = Dense(500, dropout=0.2)
        self.m4Dens2 = Dense(250, dropout_exist=False)

        self.Booster1 =  Dense(250, dropout=0)

        self.m5Conv1 = Conv(6, 0.2)
        self.m5Pool1 = tf.keras.layers.MaxPool1D()
        self.m5Conv2 = Conv(12, 0.2)
        self.m5Pool2 = tf.keras.layers.MaxPool1D()
        self.m5Conv3 = Conv(24, 0.2)
        self.m5Pool3 = tf.keras.layers.AvgPool1D()
        self.m5Dense1 = Dense(320, dropout=0.1)
        self.m5Dense2 = Dense(250, dropout_exist=False)

        self.kernel1 = self.add_weight("kernel1", shape=[1], trainable=True)
        self.kernel2 = self.add_weight("kernel2", shape=[1], trainable=True)
        self.kernel3 = self.add_weight("kernel3", shape=[1], trainable=True)
        self.kernel4 = self.add_weight("kernel4", shape=[1], trainable=True)
        self.kernel5 = self.add_weight("kernel5", shape=[1], trainable=True)
        self.kernel7 = self.add_weight("kernel7", shape=[1], trainable=True)
        self.add = tf.keras.layers.Add()

    def call(self, in_put):
        x = self.Conv1(in_put)
        x = self.Indentity1(x)
        x = self.Conv2(x)
        x = self.Indentity2(x)
        x = self.Conv3(x)
        x = self.Indentity3(x)
        x = self.Conv4(x)
        x = self.Indentity4(x)
        x = self.Flatten(x)
        x1 = self.Dense(x)

        x = self.m2Conv1(in_put)
        x = self.m2Indentity1(x)
        x = self.m2Conv2(x)
        x = self.m2Indentity2(x)
        x = self.m2Conv3(x)
        x = self.m2Indentity3(x)
        x = self.Flatten(x)
        x = self.m2Dense1(x)
        x2 = self.m2Dense2(x)

        x = self.m21Conv1(in_put)
        x = self.m21Indentity1(x)
        x = self.m21Conv2(x)
        x = self.m21Indentity2(x)
        x = self.m21Conv3(x)
        x = self.m21Indentity3(x)
        x = self.m21Conv4(x)
        x = self.Flatten(x)
        x3 = self.m21Dense(x)

        x = self.m8Conv1(in_put)
        x = self.m8Pool1(x)
        x = self.m8Conv2(x)
        x = self.m8Pool2(x)
        x = self.m8Conv3(x)
        x = self.m8Pool3(x)
        x = self.Flatten(x)
        x = self.m8Dense1(x)
        x8 = self.m8Dense2(x)


        x = self.m7Conv1(in_put)
        x = self.m7Pool1(x)
        x = self.m7Conv2(x)
        x = self.m7Pool2(x)
        x = self.m7Conv3(x)
        x = self.m7Pool3(x)
        x = self.Flatten(x)
        x = self.m7Dense1(x)
        x7 = self.m7Dense2(x)

        x = self.m5Conv1(in_put)
        x = self.m5Pool1(x)
        x = self.m5Conv2(x)
        x = self.m5Pool2(x)
        x = self.m5Conv3(x)
        x = self.m5Pool3(x)
        x = self.Flatten(x)
        x = self.m5Dense1(x)
        x5 = self.m5Dense2(x)

        results = tf.concat([x1, x2, x3,x5,x7,x8], axis=1)
        pad = self.Booster1(results)
        in_put = tf.keras.layers.Flatten()(in_put)
        inputs = tf.concat([in_put, tf.reshape(pad, [-1, 250])], axis=1)

        x = self.m3Dens1(inputs)
        x = self.m3Dens2(x)
        x4 = self.m3Dens3(x)

        k1 = self.kernel1 * x1
        k2 = self.kernel2 * x2
        k3 = self.kernel3 * x3
        k4 = self.kernel4 * x4
        k5 = self.kernel5 * x5
        k7 = self.kernel7 * x7
        x = self.add([k1, k2, k3, k4, k5, k7])

        return self.softmax(x)


import gc

warnings.filterwarnings(action='ignore')
if __name__ == '__main__':
    EPOCHS = 150
    BATCH_SIZE = 16
    datax = np.load("./feature_data.npy")
    datay = np.load("./feature_labels.npy")

    gc.collect()
    with tf.device('/GPU:0'):
        model = CNN_Model()
        model.build((None, 1086, 3))
        # print(model.summary())
        model.load_weights(filepath='./tf/best_model.h5')

        opt = tfa.optimizers.AdamW(weight_decay=1e-5, learning_rate=0.1 / 256 * BATCH_SIZE)
        metric_ls = [
            tf.keras.metrics.SparseCategoricalAccuracy(),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5),
        ]

        model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=[metric_ls])
        model.evaluate(datax, datay, batch_size=1024)
        tflite_conversion(model)
        callback_list = [
            tf.keras.callbacks.LearningRateScheduler(
                tf.keras.optimizers.schedules.CosineDecay(0.1 / 256 * BATCH_SIZE, decay_steps=350)),
            tf.keras.callbacks.ModelCheckpoint(
                f"./tf/best_model.h5",
                monitor="val_sparse_categorical_accuracy",
                verbose=0,
                save_best_only=True,
                save_weights_only=True,
                mode="max",
                save_freq="epoch",
            ),
        ]
        model.fit(datax, datay, validation_split=0.2, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=callback_list,
                  verbose=1)
