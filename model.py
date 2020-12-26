import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical


def traffic_sign_model(num_classes):
    model = Sequential()

    model.add(Conv2D(60, (5, 5), input_shape=(32, 32, 1), activation="relu"))
    model.add(Conv2D(60, (5, 5), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # (2,2) i.e 32x32x1 ---> 16x16x1

    model.add(Conv2D(30, (3, 3), activation="relu"))
    model.add(Conv2D(30, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(500, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    return model
