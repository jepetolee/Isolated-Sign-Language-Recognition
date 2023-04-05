from get_dataset import  *
dummy_dataset = get_train_batch_all_signs(X, y, NON_EMPTY_FRAME_IDXS)
import tensorflow_addons as tfa

def scaled_dot_product(q, k, v, softmax, attention_mask):
    # calculates Q . K(transpose)
    qkt = tf.matmul(q, k, transpose_b=True)
    # caculates scaling factor
    dk = tf.math.sqrt(tf.cast(q.shape[-1], dtype=tf.float32))
    scaled_qkt = qkt / dk
    softmax = softmax(scaled_qkt, mask=attention_mask)

    z = tf.matmul(softmax, v)
    # shape: (m,Tx,depth), same shape as q,k,v
    return z

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_of_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_of_heads = num_of_heads
        self.depth = d_model // num_of_heads
        self.wq = [tf.keras.layers.Dense(self.depth) for i in range(num_of_heads)]
        self.wk = [tf.keras.layers.Dense(self.depth) for i in range(num_of_heads)]
        self.wv = [tf.keras.layers.Dense(self.depth) for i in range(num_of_heads)]
        self.wo = tf.keras.layers.Dense(d_model)
        self.softmax = tf.keras.layers.Softmax()

    def call(self, x, attention_mask):
        multi_attn = []
        for i in range(self.num_of_heads):
            Q = self.wq[i](x)
            K = self.wk[i](x)
            V = self.wv[i](x)
            multi_attn.append(scaled_dot_product(Q, K, V, self.softmax, attention_mask))

        multi_head = tf.concat(multi_attn, axis=-1)
        multi_head_attention = self.wo(multi_head)
        return multi_head_attention


class Transformer(tf.keras.Model):
    def __init__(self, num_blocks):
        super(Transformer, self).__init__(name='transformer')
        self.num_blocks = num_blocks

    def build(self, input_shape):
        self.ln_1s = []
        self.mhas = []
        self.ln_2s = []
        self.mlps = []
        # Make Transformer Blocks
        for i in range(self.num_blocks):
            # First Layer Normalisation
            self.ln_1s.append(tf.keras.layers.LayerNormalization(epsilon=1e-6))
            # Multi Head Attention
            self.mhas.append(MultiHeadAttention(384, 8))
            # Second Layer Normalisation
            self.ln_2s.append(tf.keras.layers.LayerNormalization(epsilon=1e-6))
            # Multi Layer Perception
            self.mlps.append(tf.keras.Sequential([
                tf.keras.layers.Dense(384 * 2, activation='gelu', kernel_initializer='he_normal'),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(384, kernel_initializer='he_normal'),
            ]))

    def call(self, x, attention_mask):
        # Iterate input over transformer blocks
        for ln_1, mha, ln_2, mlp in zip(self.ln_1s, self.mhas, self.ln_2s, self.mlps):
            x1 = ln_1(x)
            attention_output = mha(x1, attention_mask)
            x2 = x1 + attention_output
            x3 = ln_2(x2)
            x3 = mlp(x3)
            x = x3 + x2

        return x


class LandmarkEmbedding(tf.keras.Model):
    def __init__(self, units, name):
        super(LandmarkEmbedding, self).__init__(name=f'{name}_embedding')
        self.units = units

    def build(self, input_shape):
        # Embedding for missing landmark in frame, initizlied with zeros
        self.empty_embedding = self.add_weight(
            name=f'{self.name}_empty_embedding',
            shape=[self.units],
            initializer=tf.keras.initializers.constant(0.0),
        )
        # Embedding
        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(self.units, name=f'{self.name}_dense_1', use_bias=False,
                                  kernel_initializer=tf.keras.initializers.glorot_uniform, activation='gelu'),
            tf.keras.layers.Dense(self.units, name=f'{self.name}_dense_2', use_bias=False,
                                  kernel_initializer='he_normal'),
        ], name=f'{self.name}_dense')

    def call(self, x):
        return tf.where(
            # Checks whether landmark is missing in frame
            tf.reduce_sum(x, axis=2, keepdims=True) == 0,
            # If so, the empty embedding is used
            self.empty_embedding,
                # Otherwise the landmark data is embedded
                self.dense(x),
            )


class CustomEmbedding(tf.keras.Model):
    def __init__(self):
        super(CustomEmbedding, self).__init__()

    def get_diffs(self, l):
        S = l.shape[2]
        other = tf.expand_dims(l, 3)
        other = tf.repeat(other, S, axis=3)
        other = tf.transpose(other, [0, 1, 3, 2])
        diffs = tf.expand_dims(l, 3) - other
        diffs = tf.reshape(diffs, [-1, cfg.INPUT_SIZE, S * S])
        return diffs

    def build(self, input_shape):
        # Positional Embedding, initialized with zeros
        self.positional_embedding = tf.keras.layers.Embedding(cfg.INPUT_SIZE + 1, 384,
                                                              embeddings_initializer=tf.keras.initializers.constant(0.0))
        # Embedding layer for Landmarks
        self.lips_embedding = LandmarkEmbedding(384, 'lips')
        self.left_hand_embedding = LandmarkEmbedding(384, 'left_hand')
        self.right_hand_embedding = LandmarkEmbedding(384, 'right_hand')
        self.pose_embedding = LandmarkEmbedding(384, 'pose')
        # Landmark Weights
        self.landmark_weights = tf.Variable(tf.zeros([4], dtype=tf.float32), name='landmark_weights')
        # Fully Connected Layers for combined landmarks
        self.fc = tf.keras.Sequential([
            tf.keras.layers.Dense(384, name='fully_connected_1', use_bias=False,
                                  kernel_initializer=tf.keras.initializers.glorot_uniform, activation='gelu'),
            tf.keras.layers.Dense(384, name='fully_connected_2', use_bias=False, kernel_initializer='he_normal'),
        ], name='fc')

    def call(self, lips0, left_hand0, right_hand0, pose0, non_empty_frame_idxs, training=False):
        # Lips
        lips_embedding = self.lips_embedding(lips0)
        # Left Hand
        left_hand_embedding = self.left_hand_embedding(left_hand0)
        # Right Hand
        right_hand_embedding = self.right_hand_embedding(right_hand0)
        # Pose
        pose_embedding = self.pose_embedding(pose0)
        # Merge Embeddings of all landmarks with mean pooling
        x = tf.stack((lips_embedding, left_hand_embedding, right_hand_embedding, pose_embedding), axis=3)
        # Merge Landmarks with trainable attention weights
        x = x * tf.nn.softmax(self.landmark_weights)
        x = tf.reduce_sum(x, axis=3)
        # Fully Connected Layers
        x = self.fc(x)
        # Add Positional Embedding
        normalised_non_empty_frame_idxs = tf.where(
            tf.math.equal(non_empty_frame_idxs, -1.0),
            cfg.INPUT_SIZE,
            tf.cast(
                non_empty_frame_idxs / tf.reduce_max(non_empty_frame_idxs, axis=1, keepdims=True) * cfg.INPUT_SIZE,
                tf.int32,
            ),
        )
        x = x + self.positional_embedding(normalised_non_empty_frame_idxs)

        return x


def get_model():
    # Inputs
    frames = tf.keras.layers.Input([32, N_COLS,  3], dtype=tf.float32, name='frames')
    non_empty_frame_idxs = tf.keras.layers.Input([32], dtype=tf.float32, name='non_empty_frame_idxs')
    # Padding Mask
    mask = tf.cast(tf.math.not_equal(non_empty_frame_idxs, -1), tf.float32)
    mask = tf.expand_dims(mask, axis=2)

    x = frames
    x = tf.slice(x, [0, 0, 0, 0], [-1, 32, N_COLS, 2])
    # LIPS
    lips = tf.slice(x, [0, 0,  0, 0], [-1, 32, 40, 2])
    lips = tf.where(
        tf.math.equal(lips, 0.0),
        0.0,
        (lips - LIPS_MEAN) / LIPS_STD,
    )
    lips = tf.reshape(lips, [-1, 32, 40 * 2])
    # LEFT HAND
    left_hand = tf.slice(x, [0, 0, 40, 0], [-1, 32, 21, 2])
    left_hand = tf.where(
        tf.math.equal(left_hand, 0.0),
        0.0,
        (left_hand - LEFT_HANDS_MEAN) / LEFT_HANDS_STD,
    )
    left_hand = tf.reshape(left_hand, [-1, 32, 21 * 2])
    # RIGHT HAND
    right_hand = tf.slice(x, [0, 0, 61, 0], [-1, 32, 21, 2])
    right_hand = tf.where(
        tf.math.equal(right_hand, 0.0),
        0.0,
        (right_hand - RIGHT_HANDS_MEAN) / RIGHT_HANDS_STD,
    )
    right_hand = tf.reshape(right_hand, [-1, 32, 21 * 2])
    # POSE
    pose = tf.slice(x, [0, 0, 82, 0], [-1, 32, 10, 2])
    pose = tf.where(
        tf.math.equal(pose, 0.0),
        0.0,
        (pose - POSE_MEAN) / POSE_STD,
    )
    pose = tf.reshape(pose, [-1, 32, 10 * 2])
    x = lips, left_hand, right_hand, pose
    x = CustomEmbedding()(lips, left_hand, right_hand, pose, non_empty_frame_idxs)
    # Encoder Transformer Blocks
    x = Transformer(2)(x, mask)
    # Pooling
    x = tf.reduce_sum(x * mask, axis=1) / tf.reduce_sum(mask, axis=1)
    # Classification Layer
    x = tf.keras.layers.Dense(cfg.NUM_CLASSES, activation=tf.keras.activations.softmax,
                              kernel_initializer=tf.keras.initializers.glorot_uniform)(x)
    outputs = x
    # Create Tensorflow Model
    model = tf.keras.models.Model(inputs=[frames, non_empty_frame_idxs], outputs=outputs)

    # Simple Categorical Crossentropy Loss
    loss = tf.keras.losses.SparseCategoricalCrossentropy()

    # Adam Optimizer with weight decay
    optimizer = tfa.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-5, clipnorm=1.0)

    lr_metric = get_lr_metric(optimizer)
    metrics = ["acc", lr_metric]
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model