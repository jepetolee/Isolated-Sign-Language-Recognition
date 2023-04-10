from Transformer import Transformer
from ModelFactory.BaseModels import *



def GetLinear():
    input = tf.keras.layers.Input([1932, 3], dtype=tf.float32, name='inputT')
    x = tf.transpose(input, perm=[0, 2, 1])
    x = Dense(644, dropout=0.1)(x)
    x = Dense(322, dropout=0.1)(x)
    x = tf.keras.layers.Flatten()(x)
    x = Dense(500, dropout=0.1)(x)
    x = Dense(250, dropout_exist=False)(x)
    output = tf.keras.layers.Softmax()(x)
    model = tf.keras.models.Model(inputs=input, outputs=output)
    return model

def GetTransformer():
    input = tf.keras.layers.Input([1932, 3], dtype=tf.float32, name='inputT')
    x = Conv(8, 0.1, stride=3)(input)
    x = Identity_Block(8,0.1)(x)
    x = tf.transpose(x, perm=[0, 2, 1])
    x = Transformer(num_blocks=2, units=322)(x)
    x = Dense(100, dropout=0.1)(x)
    x = tf.keras.layers.Flatten()(x)
    x = Dense(500, dropout=0.1)(x)
    x = Dense(250, dropout_exist=False)(x)
    output = tf.keras.layers.Softmax()(x)
    model = tf.keras.models.Model(inputs=input, outputs=output)
    return model

def GetGRU():
    input = tf.keras.layers.Input([1932, 3], dtype=tf.float32, name='inputT')
    x = Conv(8, 0.1, stride=3)(input)
    x = Identity_Block(8,0.1)(x)
    x = tf.transpose(x, perm=[0, 2, 1])
    x = GRUModel(units=322,dropout=0.1,num_blocks=4)(x)
    x = Dense(100, dropout=0.1)(x)
    x = tf.keras.layers.Flatten()(x)
    x = Dense(500, dropout=0.1)(x)
    x = Dense(250, dropout_exist=False)(x)
    output = tf.keras.layers.Softmax()(x)
    model = tf.keras.models.Model(inputs=input, outputs=output)
    return model

def GetConv():
    input = tf.keras.layers.Input([1932, 3], dtype=tf.float32, name='inputT')
    x = Conv(16, 0.1)(input)
    x = Identity_Block(16, 0.1)(x)
    x = tf.transpose(x, perm=[0, 2, 1])
    x = Dense(242,0.1)(x)
    x = tf.transpose(x, perm=[0, 2, 1])
    x =  Conv(32, 0.1)(x)
    x = Identity_Block(32, 0.1)(x)
    x = tf.keras.layers.Flatten()(x)
    x = Dense(960, dropout=0.1)(x)
    x = Dense(500, dropout=0.1)(x)
    x = Dense(250, dropout_exist=False)(x)
    output = tf.keras.layers.Softmax()(x)
    model = tf.keras.models.Model(inputs=input, outputs=output)
    return model


class FinalModel(tf.keras.Model):
    def __init__(self,transformer,gru,cnn,linear):
        super(FinalModel, self).__init__()
        self.transformer = transformer
        self.linear = linear
        self.cnn = cnn
        self.gru =gru

        self.kernel1 = self.add_weight("kernel1", shape=[1], trainable=True)
        self.kernel2 = self.add_weight("kernel2", shape=[1], trainable=True)
        self.kernel3 = self.add_weight("kernel3", shape=[1], trainable=True)
        self.kernel4 = self.add_weight("kernel4", shape=[1], trainable=True)
        self.add = tf.keras.layers.Add()

    def call(self, input):

        x1 = self.transformer(input)
        x2 = self.gru(input)
        x3 = self.cnn(input)
        x4 =self.linear(input)

        k1 = self.kernel1 * x1
        k2 = self.kernel2 * x2
        k3 = self.kernel3 * x3
        k4 = self.kernel4 * x4
        return self.add([k1, k2, k3, k4])

