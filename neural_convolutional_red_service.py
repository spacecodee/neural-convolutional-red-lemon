import tensorflow as tf
import os
import numpy as np
from tensorflow.keras import datasets, layers, models, Sequential
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers


# from tensorflow.keras.models import Sequential

def get_path_to_train():
    # Get the directory containing the current script
    path_to_train = os.path.dirname(os.path.realpath(filename='train/'))

    return path_to_train


# Let's train the model
def lemon_models_to_train():
    data_to_take = 0.1
    train_ds = None
    try:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            get_path_to_train(),
            validation_split=data_to_take,
            subset="training",
            seed=123,
            image_size=(300, 300))
    except FileNotFoundError as e:
        print('Error: ' + e.strerror)

    return train_ds


def lemon_models_to_validate():
    data_to_take = 0.8
    val_ds = tf.keras.utils.image_dataset_from_directory(
        get_path_to_train(),
        validation_split=data_to_take,
        subset="validation",
        seed=123,
        image_size=(300, 300))
    return val_ds


def show_images_from_train():
    train_ds = lemon_models_to_train()
    class_names = train_ds.class_names

    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            _ = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")


def validate_train_dimensions():
    train_ds = lemon_models_to_train()
    for image_batch, labels_batch in train_ds:
        print(image_batch.shape)
        print(labels_batch.shape)
        break


def build_first_model_to_train():
    train_ds = lemon_models_to_train()
    class_names = train_ds.class_names
    num_classes = len(class_names)

    model = Sequential([
        keras.Input(shape=(300, 300, 3)),  # Input layer with shape 300x300 pixels and 3 channels (RGB)
        layers.Rescaling(1. / 255),  # Normalize pixel values to be between 0 and 1 (from 0 to 255)
        layers.Conv2D(16, 3, padding='same', activation='relu'),  # 16 filters, 3x3 kernel
        layers.MaxPooling2D(),  # 2x2 pool size
        layers.Conv2D(32, 3, padding='same', activation='relu'),  # 32 filters, 3x3 kernel
        layers.MaxPooling2D(),  # 2x2 pool size
        layers.Conv2D(64, 3, padding='same', activation='relu'),  # 64 filters, 3x3 kernel
        layers.MaxPooling2D(),  # 2x2 pool size
        layers.Flatten(),  # Flatten the output of the last Conv2D layer
        layers.Dense(units=128, activation='relu'),  # 128 neurons in the Dense layer
        layers.Dense(num_classes)  # Number of classes
    ])

    # compile the model
    return model


def compile_model_to_train():
    model = build_first_model_to_train()
    model.compile(optimizer='adam',
                  # Optimizer function: Adam (Adaptive Moment Estimation) optimizer is an extension
                  # to stochastic gradient descent
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  # Loss function for the model to minimize during optimization (cross-entropy)
                  metrics=['accuracy'])  # Metric to monitor during training and testing (accuracy)

    model.summary()
    return model


def train_lemon_model():
    model = compile_model_to_train()
    train_ds = lemon_models_to_train()
    val_ds = lemon_models_to_validate()
    epochs = 4
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    return history, epochs


def show_metrics_about_train():
    epochs = train_lemon_model()[1]
    history = train_lemon_model()[0]

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


if __name__ == '__main__':
    show_metrics_about_train()
