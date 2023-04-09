import tensorflow as tf
from Transformer.Attentionlayer import MultiHeadAttention


class Transformer(tf.keras.Model):
    def __init__(self, num_blocks,units):
        super(Transformer, self).__init__()
        self.num_blocks = num_blocks
        self.UNITS = units

    def build(self, input_shape):
        self.ln_1s = []
        self.mhas = []
        self.ln_2s = []
        self.mlps = []
        # Make Transformer Blocks
        for i in range(self.num_blocks):
            # First Layer Normalisation
            self.ln_1s.append(tf.keras.layers.LayerNormalization(epsilon=1e-3))
            # Multi Head Attention
            self.mhas.append(MultiHeadAttention(self.UNITS, 8))
            # Second Layer Normalisation
            self.ln_2s.append(tf.keras.layers.LayerNormalization(epsilon=1e-3))
            # Multi Layer Perception
            self.mlps.append(tf.keras.Sequential([
                tf.keras.layers.Dense(self.UNITS * 2, activation='gelu', kernel_initializer='he_normal'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(self.UNITS, kernel_initializer='he_normal'),
                tf.keras.layers.BatchNormalization()
            ]))

    def call(self, x):
        # Iterate input over transformer blocks
        for ln_1, mha, ln_2, mlp in zip(self.ln_1s, self.mhas, self.ln_2s, self.mlps):
            x1 = ln_1(x)
            attention_output = mha(x1)
            x2 = x1 + attention_output
            x3 = ln_2(x2)
            x3 = mlp(x3)
            x = x3 + x2
        return x