import tensorflow as tf
from CNN_Model.BaseModels import Dense
def scaled_dot_product(q, k, v, softmax):
    # calculates Q . K(transpose)
    qkt = tf.matmul(q, k, transpose_b=True)
    # caculates scaling factor
    dk = tf.math.sqrt(tf.cast(q.shape[-1], dtype=tf.float32))
    scaled_qkt = qkt / dk
    softmax = softmax(scaled_qkt)

    z = tf.matmul(softmax, v)
    # shape: (m,Tx,depth), same shape as q,k,v
    return z


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_of_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_of_heads = num_of_heads
        self.depth = d_model // num_of_heads
        self.wq = [Dense(self.depth,0) for i in range(num_of_heads)]
        self.wk = [Dense(self.depth,0) for i in range(num_of_heads)]
        self.wv = [Dense(self.depth,0)for i in range(num_of_heads)]
        self.wo = Dense(d_model,0)
        self.softmax = tf.keras.layers.Softmax()

    def call(self, x):
        multi_attn = []
        for i in range(self.num_of_heads):
            Q = self.wq[i](x)
            K = self.wk[i](x)
            V = self.wv[i](x)
            multi_attn.append(scaled_dot_product(Q, K, V, self.softmax))

        multi_head = tf.concat(multi_attn, axis=-1)
        multi_head_attention = self.wo(multi_head)
        return multi_head_attention