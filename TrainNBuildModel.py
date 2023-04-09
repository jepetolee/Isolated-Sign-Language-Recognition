import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import warnings
from CNN_Model.CNNModel import CNN_Model
from Transformer.TransformerModel import Transformer


def TrainCNN():
    model = CNN_Model()
    model.build((None, 1932, 3))
    print(model.summary())
    model.load_weights(filepath='saved_model/best_cnn_model.h5')
    #model.save_weights(filepath='saved_model/best_cnn_model.h5')
    # model.evaluate(datax, datay, batch_size=1024)
    # tflite_conversion(model)
    opt = tfa.optimizers.AdamW(weight_decay=1e-5, learning_rate=0.01 / 256 * BATCH_SIZE)
    metric_ls = [
        tf.keras.metrics.SparseCategoricalAccuracy(),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5),
    ]

    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=[metric_ls])
    callback_list = [
      
        tf.keras.callbacks.ModelCheckpoint(
            f"saved_model/best_cnn_model.h5",
            monitor="val_sparse_categorical_accuracy",
            verbose=0,
            save_best_only=True,
            save_weights_only=True,
            mode="max",
            save_freq="epoch",
        )
    ]
    model.fit(datax, datay, validation_split=0.2, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=callback_list,validation_batch_size=512,
              verbose=1)
import gc
warnings.filterwarnings(action='ignore')
import os

if __name__ == '__main__':


    BATCH_SIZE = 256
    EPOCHS=150
    datax = np.load("GetData/feature_data.npy")
    datay = np.load("GetData/feature_labels.npy")
    gc.collect()
    with tf.device('/GPU:0'):
        TrainCNN()

