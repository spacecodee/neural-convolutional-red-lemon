import tensorflow as tf
import os
import numpy as np
from tensorflow.keras import Sequential, layers
import matplotlib.pyplot as plt

from tensorflow import keras


# This was made by Spacecodee

# Este método devuelve la ruta del directorio actual
def get_model_path():
    # Get the directory containing the current script
    path_to_train = os.path.dirname(os.path.realpath(filename='/model/model'))

    return path_to_train


# Este método devuelve la ruta del directorio actual para el entrenamiento
def get_path_to_train():
    # Get the directory containing the current script
    path_to_train = os.path.dirname(os.path.realpath(filename='train/train'))

    return path_to_train


# Este método devuelve la ruta del directorio actual para las pruebas
def get_path_to_test():
    # Get the directory containing the current script
    path_to_train = os.path.dirname(os.path.realpath(filename='test/test'))

    return path_to_train


# Este método prepara los datos para el entrenamiento
def lemon_models_to_train():
    data_to_take = 0.01  # Tomará el 1% de los datos
    train_ds = None  # Variable que almacenará los datos de entrenamiento
    try:
        train_ds = tf.keras.utils.image_dataset_from_directory(  # Carga los datos de entrenamiento
            get_path_to_train(),
            validation_split=data_to_take,  # Tomará el 1% de los datos para el entrenamiento
            subset="training",  # El subconjunto de datos a utilizar (entrenamiento)
            seed=123,
            # El valor de la semilla para la aleatoriedad (123) (la misma semilla siempre producirá la misma salida)
            image_size=(300, 300))  # Tamaño de la imagen (300x300 píxeles)
    except FileNotFoundError as e:
        print('Error: ' + e.strerror)

    return train_ds


# Este método prepara los datos para la validación
def lemon_models_to_validate():
    data_to_validate = 0.99  # Tomará el 99% de los datos
    val_ds = tf.keras.utils.image_dataset_from_directory(
        get_path_to_train(),
        validation_split=data_to_validate,  # Tomará el 99% de los datos para la
        subset="validation",  # El subconjunto de datos a utilizar (validación)
        seed=123,
        # El valor de la semilla para la aleatoriedad (123) (la misma semilla siempre producirá la misma salida)
        image_size=(300, 300))  # Tamaño de la imagen (300x300 píxeles)
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


# Este método construye el modelo de la red neuronal convolucional
def build_first_model_to_train():
    train_ds = lemon_models_to_train()
    class_names_out = train_ds.class_names
    num_classes = len(class_names_out)

    # aquí se construye el modelo de la red neuronal convolucional
    model = Sequential([
        keras.Input(shape=(300, 300, 3)),  # Entrada de la red neuronal convolucional (300x300 píxeles, 3 canales)
        layers.Rescaling(1. / 255),  # Reescalado de los píxeles de la imagen (valores entre 0 y 1) (normalización)

        layers.Conv2D(8, 3, padding='same', activation='relu'),
        # 8 filtros, 3x3 kernel, activación relu la cual ayuda a la red a aprender patrones no lineales en los datos
        layers.MaxPooling2D(),  # 2x2 pool size (se reduce la dimensión de la imagen y se mantiene la información)

        layers.Conv2D(16, 3, padding='same', activation='relu'),  # 16 filtros, 3x3 kernel
        layers.MaxPooling2D(),  # 2x2 pool size (se reduce la dimensión de la imagen y se mantiene la información)

        layers.Conv2D(32, 3, padding='same', activation='relu'),  # 32 filtros, 3x3 kernel
        layers.MaxPooling2D(),  # 2x2 pool size (se reduce la dimensión de la imagen y se mantiene la información)

        layers.Conv2D(64, 3, padding='same', activation='relu'),  # 64 filtros, 3x3 kernel
        layers.MaxPooling2D(),  # 2x2 pool size (se reduce la dimensión de la imagen y se mantiene la información)

        layers.Conv2D(128, 3, padding='same', activation='relu'),  # 128 filtros, 3x3 kernel
        layers.MaxPooling2D(),  # 2x2 pool size (se reduce la dimensión de la imagen y se mantiene la información)

        layers.Flatten(),  # Aplanamiento de la imagen (se convierte en un vector unidimensional)
        layers.Dropout(0.5),  # Dropout layer with a rate of 0.5 (half of the input units are dropped)
        # Dropout sirve para evitar el sobre ajuste en la red neuronal y mejorar la generalización

        layers.Dense(units=128, activation='relu'),
        # 128 neuronas en la capa densa para aprender patrones, activación relu
        layers.Dense(num_classes)  # Número de clases en la capa densa
    ])

    # compile the model
    return model


# Este método compila el modelo para el entrenamiento
def compile_model_to_train():
    model = build_first_model_to_train()  # Obtener el modelo de la red neuronal convolucional
    model.compile(optimizer='adam',
                  # Adam es un algoritmo de optimización que se puede utilizar para entrenar redes neuronales
                  # y minimizar la función de pérdida
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  # Función de pérdida para medir qué tan bien el modelo se ajusta a los datos de entrenamiento
                  metrics=['accuracy'])  # Métrica para evaluar el rendimiento del modelo (precisión)

    model.summary()  # Muestra un resumen del modelo de la red neuronal convolucional construido
    return model


# Es momento de entrenar el modelo
def train_lemon_model():
    model = compile_model_to_train()
    train_ds = lemon_models_to_train()
    val_ds = lemon_models_to_validate()
    epochs = 20  # Número de épocas para entrenar el modelo
    history = model.fit(  # Entrenamiento del modelo
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    return history, epochs, model


# Este método muestra las métricas del entrenamiento
def show_metrics_about_train():
    history = train_lemon_model()[0]  # Obtener el historial del entrenamiento
    epochs = train_lemon_model()[1]  # Obtener el número de épocas

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)  # Rango de épocas

    # Gráficas de las métricas del entrenamiento
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    # Construir la gráfica de la pérdida
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


# Este método guarda el modelo entrenado
def save_model():
    show_metrics_about_train()
    model = train_lemon_model()[2]
    model.save('model/neural_c_n_lemon_model.keras')


# Este método carga el modelo entrenado
def load_model():
    extension = '.keras'
    # save_model()
    model = keras.models.load_model('model/neural_c_n_lemon_model' + extension)
    return model


# Luego de entrenar el modelo, es momento de probar la probabilidad de éxito de este.
def try_with_saved_model():
    # modelos a probar
    model_1 = '/mycosphaerella_citri/'
    model_2 = '/planococcus_citri/'
    model_3 = '/tetranychus_urticae/'
    model_4 = '/aphididae/'
    model_5 = '/phyllocnistis_citrella/'

    file_name = 'sheet-test-'
    file_extension = '.jpg'

    class_names_out = lemon_models_to_train().class_names
    model = load_model()
    image_path = get_path_to_test() + model_5 + file_name + '3' + file_extension

    # Cargar la imagen para probar
    image = tf.keras.preprocessing.image.load_img(image_path)
    input_arr = tf.keras.preprocessing.image.img_to_array(image)  # Convertir la imagen en un array
    input_arr = np.array([input_arr])  # Convertir el array en un array de numpy
    predictions = model.predict(input_arr)  # Predecimos a qué enfermedad le pertenece la imagen con la CNN

    score = tf.nn.softmax(predictions[0])

    # make a loop of predictions values
    for i in range(0, len(score)):
        print(
            "This image looks like: {} with a percentage {:.2f} % of similarity."
            .format(class_names_out[i], 100 * score[i])
        )
    print('\n')
    print('The most likely class is: ' + class_names_out[np.argmax(score)])
    print('\n')
    print(
        "This image seems to be {} with a percentage {:.2f} % of similarity."
        .format(class_names_out[np.argmax(score)], 100 * np.max(score))
    )


if __name__ == '__main__':
    class_names = lemon_models_to_train().class_names
    print('these are the classes: \n')
    print(class_names)
    print('\n')
    try_with_saved_model()
