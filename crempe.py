# import os

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Reshape,
    BatchNormalization,
    MaxPooling2D,
    Dropout,
    Permute,
    Flatten,
    Dense
)


def crempe() -> Model:
    # Load CREPE model and return the flattened layer
    crepe_model = load_crepe_model()
    crepe_input = crepe_model.input
    x = crepe_model.layers[-2].output

    # New layer to increase feature space from 2048 (CREPE) to 4096
    y = Dense(4096, activation='relu', name="feature-space")(x)

    # New output to increase bins resolution:
    # 6 octaves * 12 semitones per octave * 10 bins per semitone = 720 bins
    # 100 cents per semitone / 10 bins per semitone = 10 cents per bin
    y = Dense(720, activation='sigmoid', name="crempe-classifier")(y)

    model = Model(inputs=crepe_input, outputs=y)

    # Freeze CREPE model weights
    for layer in model.layers[:1]:  # First test: freeze only weights of first layer
        layer.trainable = False

    return model


def crepe() -> Model:
    layers = [1, 2, 3, 4, 5, 6]
    filters = [1024, 128, 128, 128, 256, 512]
    filters_sizes = [512, 64, 64, 64, 64, 64]
    strides = [(4, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]

    x = Input(shape=(1024,), name='input', dtype='float32')
    y = Reshape(target_shape=(1024, 1, 1), name='input-reshape')(x)

    for layer, filters, filter_size, strides in zip(layers, filters, filters_sizes, strides):
        y = Conv2D(filters, (filter_size, 1), strides=strides, padding='same', activation='relu', name=f"conv{layer}")(y)
        y = BatchNormalization(name=f"conv{layer}-BN")(y)
        y = MaxPooling2D(pool_size=(2, 1), strides=None, padding='valid', name=f"conv{layer}-maxpool")(y)
        y = Dropout(0.25, name=f"conv{layer}-dropout")(y)

    y = Permute((2, 1, 3), name="transpose")(y)
    y = Flatten(name="flatten")(y)
    y = Dense(360, activation='sigmoid', name="crepe-classifier")(y)

    model = Model(inputs=x, outputs=y)

    return model


def load_crepe_model() -> Model:
    # package_dir = os.path.dirname(os.path.realpath(__file__))
    filename = "model-full.h5"
    # filepath = os.path.join(package_dir, filename)
    model = crepe()
    model.load_weights(filename)

    return model


if __name__ == '__main__':
    crempe_model = crempe()
    crempe_model.summary()