import pickle
import random
import os
import argparse

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2

from utils import load_dataset, preprocess, save_model
from model import traffic_sign_model


def main(args):

    learning_rate = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    steps_per_epoch = args.spe
    data_path = args.data_dir
    path = args.save_dir
    model_name = "traffic_signs_model"
    num_classes = 43

    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(data_path)
    data = pd.read_csv(os.path.join(data_path, "signnames.csv"))

    X_train = np.array(list(map(preprocess, X_train)))
    X_val = np.array(list(map(preprocess, X_val)))
    X_test = np.array(list(map(preprocess, X_test)))

    X_train = X_train.reshape(34799, 32, 32, 1)
    X_val = X_val.reshape(4410, 32, 32, 1)
    X_test = X_test.reshape(12630, 32, 32, 1)

    datagen = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        shear_range=0.15,
        rotation_range=30,
    )
    datagen.fit(X_train)

    y_train = to_categorical(y_train, 43)
    y_val = to_categorical(y_val, 43)
    y_test = to_categorical(y_test, 43)

    model = traffic_sign_model(num_classes)
    model.compile(
        Adam(lr=learning_rate), loss="categorical_crossentropy", metrics=["accuracy"]
    )

    print("[INFOS]Starting training...")
    history = model.fit_generator(
        datagen.flow(X_train, y_train, batch_size),
        steps_per_epoch,
        epochs,
        validation_data=(X_val, y_val),
        shuffle=1,
        verbose=1,
    )
    save_model(model, model_name, path="/saved")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="command line for different parameters"
    )
    parser.add_argument(
        "-data_dir",
        type=str,
        required=True,
        default="./data",
        help="/path/to/german-traffic-sign-dataset",
    )
    parser.add_argument(
        "-lr",
        type=float,
        required=False,
        default=0.00099,
        help="learning rate for the Adam ex : 0.01, 0,001 . default : 0.00099",
    )
    parser.add_argument(
        "-batch_size",
        type=int,
        required=False,
        default=32,
        help="how many samples per batch. default : 32",
    )
    parser.add_argument(
        "-epochs",
        type=int,
        required=False,
        default=10,
        help="number of epochs. default : 10",
    )
    parser.add_argument(
        "-spe",
        type=int,
        required=False,
        default=600,
        help="steps per epoch. default : 600",
    )
    parser.add_argument(
        "-save_dir",
        type=str,
        required=False,
        help="directory to save the model. default: ./saved",
        default="./saved",
    )

    args = parser.parse_args()

    main(args)
