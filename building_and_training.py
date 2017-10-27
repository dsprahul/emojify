import os
import cv2
import keras
import numpy as np
from face_features import FaceFeatures
from keras.layers import Dense, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model

BATCH_SIZE = 32


def convolution_stack(input_layer):

    c1 = Conv2D(filters=16, kernel_size=(3, 3), activation="relu",
                padding="same")(input_layer)
    c1 = Conv2D(filters=16, kernel_size=(3, 3), activation="relu",
                padding="same")(c1)
    c1 = MaxPooling2D((2, 2), strides=(2, 2))(c1)

    c2 = Conv2D(filters=64, kernel_size=(3, 3), activation="relu",
                padding="same")(c1)
    c2 = Conv2D(filters=16, kernel_size=(3, 3), activation="relu",
                padding="same")(c2)
    c2 = MaxPooling2D((2, 2), strides=(2, 2))(c2)

    c3 = Conv2D(filters=128, kernel_size=(3, 3), activation="relu",
                padding="same")(c2)
    c3 = Conv2D(filters=16, kernel_size=(3, 3), activation="relu",
                padding="same")(c3)
    c3 = MaxPooling2D((2, 2), strides=(2, 2))(c3)

    c4 = Conv2D(filters=256, kernel_size=(3, 3), activation="relu",
                padding="same")(c3)
    c4 = Conv2D(filters=16, kernel_size=(3, 3), activation="relu",
                padding="same")(c4)
    c4 = MaxPooling2D((2, 2), strides=(2, 2))(c4)

    flattened = Flatten()(c4)

    fc1 = Dense(output_dim=256, activation="relu")(flattened)
    partial_model_output_layer = Dense(output_dim=64, activation="relu")(fc1)

    return partial_model_output_layer


def build_hierarchial_model():

    eyes_input = Input(shape=(400, 16, 1), name="eyes_input")
    eyes_model = convolution_stack(input_layer=eyes_input)

    nose_input = Input(shape=(400, 16, 1), name="nose_input")
    nose_model = convolution_stack(input_layer=nose_input)

    mouth_input = Input(shape=(400, 16, 1), name="mouth_input")
    mouth_model = convolution_stack(input_layer=mouth_input)

    layer_summation = keras.layers.concatenate([eyes_model,
                                                nose_model,
                                                mouth_model])

    fully_connected = Dense(output_dim=32, activation="relu",
                            name="fc1")(layer_summation)
    output_layer = Dense(output_dim=5, activation="softmax",
                         name="fc2")(fully_connected)

    model = Model(inputs=[eyes_input, nose_input, mouth_input],
                  output=output_layer)

    # print model.summary()
    return model


if __name__ == "__main__":

    # model = build_hierarchial_model()
    # model.compile(optimizer="adam", loss="categorical_crossentropy")

    xs, ys = load_training_data()
    # model.fit(x=xs, y=ys, nb_epoch=1, batch_size=16)
