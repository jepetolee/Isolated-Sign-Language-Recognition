
import tensorflow as tf
class Conv(tf.keras.layers.Layer):
    def __init__(self, num_outputs, dropout,stride=2):
        super(Conv, self).__init__()

        self.conv1 = tf.keras.layers.Conv1D(num_outputs, kernel_size=2, strides=1,
                                            kernel_initializer='he_normal')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.act1 = tf.keras.layers.Activation('gelu')
        self.conv2 = tf.keras.layers.Conv1D(num_outputs, kernel_size=2, strides=stride, activation='gelu',
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