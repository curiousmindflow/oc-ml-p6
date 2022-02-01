import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pickle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.xception import Xception


EXTENSIONS_ALLOWED = ["png", "jpg", "jpeg"]
CHECKPOINT_PATH = './data/saves/xception.hdf5'
BREED_LIST_PATH = "data/labels.list"


def main():
    args = sys.argv[1:]
    if len(args) == 0:
        print("There must be one arg, the path to the picture to classify")
    if len(args) > 1:
        print("Only the first arg will be used")
    model = load_model(CHECKPOINT_PATH)
    img = load_image(args[0])
    img_preproc = preprocess(img)
    prediction = predict(model, img_preproc)[0]
    breed_list = load_breeds(BREED_LIST_PATH)
    result = {k: v for (k, v) in zip(breed_list, prediction)}
    result = {k: v for k, v in sorted(
        result.items(),
        key=lambda item: item[1],
        reverse=True
        )}
    print(f">>> Result: {result} <<<")


def load_model_by_weights(checkpoint_path: str):
    try:
        os.path.exists(checkpoint_path + "index")
    except Exception:
        print(f"File not found at: {checkpoint_path}")
        exit()
    xception = Xception(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(224, 224, 3)
    )
    model = keras.Sequential([
        xception,
        layers.Dense(units=128, activation="relu"),
        layers.Dropout(rate=0.2),
        layers.Dense(units=10, activation="softmax")
    ])
    model.load_weights(checkpoint_path)
    return model


def load_model(checkpoint_path: str):
    try:
        os.path.exists(checkpoint_path)
    except Exception:
        print(f"File not found at: {checkpoint_path}")
        exit()
    model = tf.keras.models.load_model(checkpoint_path, compile=False)
    return model


def predict(model, img_preproc):
    prediction = model.predict(img_preproc)
    return prediction


def load_image(path: str):
    try:
        os.path.exists(path)
    except Exception:
        print(f"File not found at: {path}")
        exit()
    extension = path.split(".")[-1]
    if extension not in EXTENSIONS_ALLOWED:
        print(f"Extension '{extension}' not allowed")
        exit()
    image = load_img(path, target_size=(224, 224))
    return image


def preprocess(img):
    img = img_to_array(img)
    img = img.reshape((img.shape[0], img.shape[1], img.shape[2]))
    img = np.expand_dims(img, axis=0)
    img /= 255
    return img


def load_breeds(path: str):
    with open(path, "rb") as stream:
        breed_list = pickle.load(stream)
        return breed_list


if __name__ == "__main__":
    main()
