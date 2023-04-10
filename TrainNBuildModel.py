import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import warnings
from ModelFactory.MakeNTestModel import GetTransformer,GetGRU ,GetConv,GetLinear,FinalModel

def TrainGRU(BATCH_SIZE:int,EPOCHS:int,pretrained:bool):
    model = GetGRU()
    print(model.summary())
    if pretrained:
        model.load_weights(filepath='saved_model/best_GRU_model.h5')
    model.save_weights(filepath='saved_model/sizetest_GRU_model.h5')
    # model.evaluate(datax, datay, batch_size=1024)
    # tflite_conversion(model)
    opt = tfa.optimizers.AdamW(weight_decay=1e-5, learning_rate=0.01 / 256 * BATCH_SIZE)
    metric_ls = [
        tf.keras.metrics.SparseCategoricalAccuracy(),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5),
    ]

    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=[metric_ls])
    callback_list = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2,
                                             patience=4096 // BATCH_SIZE, min_lr=0.001),
        tf.keras.callbacks.ModelCheckpoint(
            f"saved_model/best_GRU_model.h5",
            monitor="val_sparse_categorical_accuracy",
            verbose=0,
            save_best_only=True,
            save_weights_only=True,
            mode="max",
            save_freq="epoch",
        ),
        tf.keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', patience=15)
    ]
    model.fit(datax, datay, validation_split=0.2, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=callback_list,
              validation_batch_size=512,
              verbose=1)
def TrainTransformer(BATCH_SIZE:int,EPOCHS:int,pretrained:bool):
    model = GetTransformer()
    print(model.summary())
    if pretrained:
     model.load_weights(filepath='saved_model/best_transformer_model.h5')
    model.save_weights(filepath='saved_model/sizetest_transformer_model.h5')

    # model.evaluate(datax, datay, batch_size=1024)
    # tflite_conversion(model)
    opt = tfa.optimizers.AdamW(weight_decay=1e-5, learning_rate=0.01 / 256 * BATCH_SIZE)
    metric_ls = [
        tf.keras.metrics.SparseCategoricalAccuracy(),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5),
    ]

    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=[metric_ls])
    callback_list = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2,
                          patience=4096//BATCH_SIZE, min_lr=0.001),
        tf.keras.callbacks.ModelCheckpoint(
            f"saved_model/best_transformer_model.h5",
            monitor="val_sparse_categorical_accuracy",
            verbose=0,
            save_best_only=True,
            save_weights_only=True,
            mode="max",
            save_freq="epoch",
        ),
        tf.keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy',patience=15)
    ]
    model.fit(datax, datay, validation_split=0.2, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=callback_list,validation_batch_size=512,
              verbose=1)


def TrainCNN(BATCH_SIZE:int,EPOCHS:int,pretrained:bool):
    model = GetConv()
    print(model.summary())
    if pretrained:
        model.load_weights(filepath='saved_model/best_CNN_model.h5')
    model.save_weights(filepath='saved_model/sizetest_CNN_model.h5')
    # model.evaluate(datax, datay, batch_size=1024)
    # tflite_conversion(model)
    opt = tfa.optimizers.AdamW(weight_decay=1e-5, learning_rate=0.01 / 256 * BATCH_SIZE)
    metric_ls = [
        tf.keras.metrics.SparseCategoricalAccuracy(),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5),
    ]

    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=[metric_ls])
    callback_list = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2,
                          patience=4096//BATCH_SIZE, min_lr=0.001),
        tf.keras.callbacks.ModelCheckpoint(
            f"saved_model/best_CNN_model.h5",
            monitor="val_sparse_categorical_accuracy",
            verbose=0,
            save_best_only=True,
            save_weights_only=True,
            mode="max",
            save_freq="epoch",
        ),
        tf.keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy',patience=15)
    ]
    model.fit(datax, datay, validation_split=0.2, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=callback_list,validation_batch_size=512,
              verbose=1)

def TrainLinear(BATCH_SIZE:int,EPOCHS:int,pretrained:bool):
    model = GetLinear()
    print(model.summary())
    if pretrained:
        model.load_weights(filepath='saved_model/best_Linear_model.h5')
    model.save_weights(filepath='saved_model/sizetest_Linear_model.h5')
    # model.evaluate(datax, datay, batch_size=1024)
    # tflite_conversion(model)
    opt = tfa.optimizers.AdamW(weight_decay=1e-5, learning_rate=0.01 / 256 * BATCH_SIZE)
    metric_ls = [
        tf.keras.metrics.SparseCategoricalAccuracy(),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5),
    ]

    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=[metric_ls])
    callback_list = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2,
                          patience=4096//BATCH_SIZE, min_lr=0.001),
        tf.keras.callbacks.ModelCheckpoint(
            f"saved_model/best_Linear_model.h5",
            monitor="val_sparse_categorical_accuracy",
            verbose=0,
            save_best_only=True,
            save_weights_only=True,
            mode="max",
            save_freq="epoch",
        ),
        tf.keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy',patience=15)
    ]
    model.fit(datax, datay, validation_split=0.2, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=callback_list,validation_batch_size=512,
              verbose=1)


def TrainFinal(BATCH_SIZE:int,EPOCHS:int,pretrained:bool,PF:bool):
    Transformer = GetTransformer()
    GRU =GetGRU()
    CNN = GetConv()
    Linear = GetLinear()


    if pretrained:
        Transformer.load_weights(filepath='saved_model/best_transformer_model.h5')
        GRU.load_weights(filepath='saved_model/best_GRU_model.h5')
        CNN.load_weights(filepath='saved_model/best_CNN_model.h5')
        Linear.load_weights(filepath='saved_model/best_Linear_model.h5')

    model = FinalModel(Transformer,GRU,CNN,Linear)
    if PF:
        model.load_weights(filepath='saved_model/best_Final_model.h5')
    model.build(input_shape=(None,1932,3))
    print(model.summary())

    model.save_weights(filepath='saved_model/sizetest_Final_model.h5')
    # model.evaluate(datax, datay, batch_size=1024)
    # tflite_conversion(model)
    opt = tfa.optimizers.AdamW(weight_decay=1e-5, learning_rate=0.01 / 256 * BATCH_SIZE)
    metric_ls = [
        tf.keras.metrics.SparseCategoricalAccuracy(),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5),
    ]

    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=[metric_ls])
    callback_list = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2,
                          patience=4096//BATCH_SIZE, min_lr=0.001),
        tf.keras.callbacks.ModelCheckpoint(
            f"saved_model/best_Final_model.h5",
            monitor="val_sparse_categorical_accuracy",
            verbose=0,
            save_best_only=True,
            save_weights_only=True,
            mode="max",
            save_freq="epoch",
        ),
        tf.keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy',patience=15)
    ]
    model.fit(datax, datay, validation_split=0.2, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=callback_list,validation_batch_size=512,
              verbose=1)

import gc
warnings.filterwarnings(action='ignore')

if __name__ == '__main__':
    datax = np.load("GetData/feature_data.npy")
    datay = np.load("GetData/feature_labels.npy")
    gc.collect()
    with tf.device('/GPU:0'):

        TrainLinear(BATCH_SIZE=16,EPOCHS=3,pretrained=False)
        TrainCNN(BATCH_SIZE=16,EPOCHS=3,pretrained=False)
        TrainTransformer(BATCH_SIZE=16,EPOCHS=3,pretrained=False)
        TrainGRU(BATCH_SIZE=16,EPOCHS=3,pretrained=False)

        TrainLinear(BATCH_SIZE=512, EPOCHS=40, pretrained=True)
        TrainCNN(BATCH_SIZE=512,EPOCHS=40,pretrained=True)
        TrainTransformer(BATCH_SIZE=512,EPOCHS=40,pretrained=True)
        TrainGRU(BATCH_SIZE=512,EPOCHS=40,pretrained=True)

        TrainFinal(BATCH_SIZE=16,EPOCHS=3,pretrained=True,PF=False)
        TrainFinal(BATCH_SIZE=512, EPOCHS=40, pretrained=False,PF=True)
