import pickle
import os

import cv2
import numpy as np
import requests
from PIL import Image


def load_dataset(path):
    with open(os.path.join(path, "train.p"), "rb") as f:
        train_data = pickle.load(f)
    with open(os.path.join(path, "valid.p"), "rb") as f:
        val_data = pickle.load(f)
    with open(os.path.join(path, "test.p"), "rb") as f:
        test_data = pickle.load(f)

    X_train, y_train = train_data["features"], train_data["labels"]
    X_val, y_val = val_data["features"], val_data["labels"]
    X_test, y_test = test_data["features"], test_data["labels"]
    return X_train, y_train, X_val, y_val, X_test, y_test


def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img


def preprocess_infer(img):
    img = np.asarray(img)
    img = cv2.resize(img, (32, 32))
    img = preprocess(img)
    return img


def fetch_img(url):
    r = requests.get(url, stream=True)
    img = Image.open(r.raw)
    return img


def save_model(model, name, path=None):

    model_path = os.path.join(path, path)
    model_name = str(name) + ".h5"
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    model.save(os.path.join(model_path, model_name))
    print("[INFOS]Model saved...")


# visualization utils
