import random
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
from Model_EX2 import get_transformer


import gc

warnings.filterwarnings(action='ignore')
if __name__ == '__main__':

    EPOCHS = 150
    BATCH_SIZE = 16
    datax = np.load("./feature_data.npy")
    datay = np.load("./feature_labels.npy")
    gc.collect()
    with tf.device('/GPU:0'):
        model = get_transformer()
        print(model.summary())
        #model.load_weights(filepath='./tf/best_model.h5')

       # model.evaluate(datax, datay, batch_size=1024)
        #tflite_conversion(model)
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