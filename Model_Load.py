import pandas as pd
from Parquet2Numpy import FeatureGen
import tensorflow as tf

ROWS_PER_FRAME = 543
import numpy as np

def load_relevant_data_subset(pq_path):
    data_columns = ["x", "y", "z"]
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)


import json


from zipfile import ZipFile

def read_json_file(file_path=f"./asl-signs/sign_to_prediction_index_map.json"):
    with open(file_path, "r") as file:
        json_data = json.load(file)
    return json_data


def tflite_conversion(model):
    inputs = tf.keras.layers.Input(shape=(543, 3))
    x = FeatureGen()(inputs)
    out = model(x)
    # TFLite Conversion
    tflite_keras_model = tf.keras.Model(inputs=inputs, outputs=out)

    keras_model_converter = tf.lite.TFLiteConverter.from_keras_model(tflite_keras_model)
    tflite_model = keras_model_converter.convert()

    tf_lite_model_path = f'./model.tflite'
    with open(tf_lite_model_path, 'wb') as f:
        f.write(tflite_model)

    ZipFile('submission.zip', mode='w').write(tf_lite_model_path)


    # print("GT   : ", train_df.sign[0])
