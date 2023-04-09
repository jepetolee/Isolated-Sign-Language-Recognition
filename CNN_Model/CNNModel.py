from Transformer import Transformer
from CNN_Model.BaseModels import *

class CNN_Model(tf.keras.Model):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.Conv1 = Conv(8, 0.1)
        self.Indentity1 = Identity_Block(8, 0.1)
        self.Conv2 = Conv(16, 0.1)
        self.Indentity2 = Identity_Block(16, 0.1)
        self.Conv3 = Conv(32, 0.1)
        self.Indentity3 = Identity_Block(32, 0.1)
        self.transc3 = Transformer(num_blocks=2, units=32)
        self.inter1 =Dense(500,0.1)
        self.Dense1 = Dense(250,dropout_exist=False)

        self.m2Conv1 = Conv(7, 0.2)
        self.m2Pool1 = tf.keras.layers.AvgPool1D(pool_size=2)
        self.m2Conv2 = Conv(14, 0.2)
        self.m2Pool2 = tf.keras.layers.AvgPool1D(pool_size=2)
        self.m2Conv3 = Conv(28, 0.2)
        self.m2Pool3 = tf.keras.layers.AvgPool1D(pool_size=2)
        self.transt3 = Transformer(num_blocks=2, units=28)
        self.inter2 = Dense(500, 0.1)
        self.Dense2 = Dense(250, dropout_exist=False)

        self.m3Conv1 = Conv(6, 0.2)
        self.m3Pool1 = tf.keras.layers.MaxPool1D(pool_size=2)
        self.m3Conv2 = Conv(12, 0.1)
        self.m3Pool2 = tf.keras.layers.MaxPool1D(pool_size=2)
        self.m3Conv3 = Conv(20, 0.1)
        self.m3Pool3 = tf.keras.layers.MaxPool1D(pool_size=2)
        self.trans3 = Transformer(num_blocks=4, units=20)
        self.Dense3 = Dense(250, dropout_exist=False)

        self.Embed = Conv(8, 0.1,stride=3)
        self.transformer1 = Transformer(num_blocks=2, units=644)
        self.TDense1 = Dense(250, dropout=0.1)
        self.TDense2 = Dense(500,dropout=0.1)
        self.Classifier = Dense(250, dropout_exist=False)


        self.Flatten = tf.keras.layers.Flatten()
        self.softmax = tf.keras.layers.Softmax(dtype="float32", name="outputs")

        self.kernel1 = self.add_weight("kernel1", shape=[1], trainable=True)
        self.kernel2 = self.add_weight("kernel2", shape=[1], trainable=True)
        self.kernel3 = self.add_weight("kernel3", shape=[1], trainable=True)
        self.kernel4 = self.add_weight("kernel4", shape=[1], trainable=True)
        self.add = tf.keras.layers.Add()

    def call(self, in_put):
        x = self.Conv1(in_put)
        x = self.Indentity1(x)
        x = self.Conv2(x)
        x = self.Indentity2(x)
        x = self.Conv3(x)
        x = self.Indentity3(x)
        x = self.transc3(x)
        x = self.Flatten(x)
        x = self.inter1(x)
        x1 = self.Dense1(x)

        x = self.m2Conv1(in_put)
        x = self.m2Pool1(x)
        x = self.m2Conv2(x)
        x = self.m2Pool2(x)
        x = self.m2Conv3(x)
        x = self.m2Pool3(x)
        x = self.transt3(x)
        x = self.Flatten(x)
        x = self.inter2(x)
        x2 = self.Dense2(x)

        x = self.m3Conv1(in_put)
        x = self.m3Pool1(x)
        x = self.m3Conv2(x)
        x = self.m3Pool2(x)
        x = self.m3Conv3(x)
        x = self.m3Pool3(x)
        x = self.trans3(x)
        x = self.Flatten(x)
        x3 = self.Dense3(x)

        x = self.Embed(in_put)
        x = tf.transpose(x,perm = [0,2,1])
        x = self.transformer1(x)
        x= self.TDense1(x)
        x = self.Flatten(x)
        x = self.TDense2(x)
        x4 = self.Classifier(x)


        k1 = self.kernel1 * x1
        k2 = self.kernel2 * x2
        k3 = self.kernel3 * x3
        k4 = self.kernel4 * x4
        x = self.add([k1, k2, k3, k4])

        return self.softmax(x)
