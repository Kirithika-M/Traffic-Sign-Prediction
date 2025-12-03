import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
from load_dataset import load_dataset_from_folders, pre_process, split_dataset


def create_model():
    model = keras.Sequential([
        ## First Convolutional Block
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),

        ## Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        ## Third Convolutional Block
        layers.Conv2D(64, (3, 3), activation="relu"),

        ## Flatten and dense layers
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(43, activation="softmax")      ## 43 classes for GTSRB
    ])

    return model


def compile_model(model):
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


if __name__ == "__main__":
    print("Creating CNN model...")

    x, y = load_dataset_from_folders()

    x, y = pre_process(x, y)

    x_train, x_val, x_test, y_train, y_val, y_test = split_dataset(x, y)

    ## Creating the model
    model = create_model()

    ## Compiling the model
    model = compile_model(model)

    ## Displaying the model architecture
    model.summary()

    print("Model created successfully!")

    history = model.fit(
        x_train, y_train, 
        batch_size=32,
        epochs=5, 
        validation_data=(x_val, y_val),
        verbose=1
    )

    test_loss, test_acc = model.evaluate(x_test, y_test)

    os.makedirs("models", exist_ok=True)
    model.save("models/traffic_sign_model.h5")
    print("Model saved!")