import tensorflow as tf
import os
import numpy as np
from tensorflow.keras import datasets, layers, models, Sequential
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers


# This was made by Spacecodee

def get_model_path():
    # Get the directory containing the current script
    path_to_train = os.path.dirname(os.path.realpath(filename='/model/model'))

    return path_to_train


def get_path_to_train():
    # Get the directory containing the current script
    path_to_train = os.path.dirname(os.path.realpath(filename='train/train'))

    return path_to_train


def get_path_to_test():
    # Get the directory containing the current script
    path_to_train = os.path.dirname(os.path.realpath(filename='test/test'))

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
            seed=123,  # Seed for random shuffling applied to the data before applying the split
            image_size=(300, 300))
    except FileNotFoundError as e:
        print('Error: ' + e.strerror)

    return train_ds


def lemon_models_to_validate():
    data_to_take = 0.9
    val_ds = tf.keras.utils.image_dataset_from_directory(
        get_path_to_train(),
        validation_split=data_to_take,
        subset="validation",
        seed=123,
        # Seed for random shuffling applied to the data before applying the split, in other words seed means that the
        # same seed will always produce the same output (in this case the same split) in the random shuffling
        image_size=(300, 300))
    return val_ds


def show_images_from_train():
    train_ds = lemon_models_to_train()
    class_names_out = train_ds.class_names

    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            _ = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names_out[labels[i]])
            plt.axis("off")


def validate_train_dimensions():
    train_ds = lemon_models_to_train()
    for image_batch, labels_batch in train_ds:
        print(image_batch.shape)
        print(labels_batch.shape)
        break


def build_first_model_to_train():
    train_ds = lemon_models_to_train()
    class_names_out = train_ds.class_names
    num_classes = len(class_names_out)

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
        layers.Dropout(0.5),  # Dropout layer with a rate of 0.5 (half of the input units are dropped)
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
    epochs = 20
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    return history, epochs, model


def show_metrics_about_train():
    history = train_lemon_model()[0]
    epochs = train_lemon_model()[1]

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


def try_the_model():
    class_names_out = lemon_models_to_train().class_names
    model = compile_model_to_train()
    image_path = get_path_to_test() + '/mycosphaerella_citri/sheet-test-1.jpg'

    image = tf.keras.preprocessing.image.load_img(image_path)
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    predictions = model.predict(input_arr)

    score = tf.nn.softmax(predictions[0])
    print(
        "Esta imagen parece ser {} con un {:.2f} % de exactitud."
        .format(class_names_out[np.argmax(score)], 100 * np.max(score))
    )


def save_model():
    show_metrics_about_train()
    model = train_lemon_model()[2]
    model.save('model/neural_c_n_lemon_model.keras')


def load_model():
    extension = '.keras'
    # save_model()
    model = keras.models.load_model('model/neural_c_n_lemon_model' + extension)
    return model


def try_with_saved_model():
    class_names_out = lemon_models_to_train().class_names
    model = load_model()
    image_path = get_path_to_test() + '/planococcus_citri/sheet-test-1.jpg'

    image = tf.keras.preprocessing.image.load_img(image_path)
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    predictions = model.predict(input_arr)

    score = tf.nn.softmax(predictions[0])

    # make a loop of predictions values
    for i in range(0, len(score)):
        print(
            "Esta imagen parece ser {} con un {:.2f} % de exactitud."
            .format(class_names_out[i], 100 * score[i])
        )
    print('\n')
    print('The most likely class is: ' + class_names_out[np.argmax(score)])
    print('\n')
    print(
        "Esta imagen parece ser {} con un {:.2f} % de exactitud."
        .format(class_names_out[np.argmax(score)], 100 * np.max(score))
    )


if __name__ == '__main__':
    class_names = lemon_models_to_train().class_names
    print('these are the classes: \n')
    print(class_names)
    try_with_saved_model()
