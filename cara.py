import os
import random
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import cv2

warnings.filterwarnings("ignore", category=UserWarning)

# Verificar las dispositivos disponibles
devices = tf.config.list_physical_devices('GPU')
if len(devices) > 0:
    print("GPU encontrada.")
    for device in devices:
        print("Dispositivo:", device)
else:
    print("No se encontró ninguna GPU.")

# Verificar si TensorFlow está utilizando la GPU por defecto
print("TensorFlow está utilizando la GPU por defecto:", tf.test.is_built_with_cuda())
print("TensorFlow está utilizando la GPU:", tf.config.list_physical_devices('GPU'))

# Declarar las dimensiones de las imágenes y el tamaño del batch
img_height, img_width = 100, 100
batch_size = 16

# Ruta del directorio de entrenamiento
train_data_dir = 'Entrenamiento'

# Obtener la lista de todas las imágenes de entrenamiento
train_image_files = []
for file in os.listdir(os.path.join(train_data_dir, 'Osvaldo')):
    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
        file_path = os.path.join(train_data_dir, 'Osvaldo', file)
        train_image_files.append(file_path)
        print("Ruta del archivo de entrenamiento:", file_path)

# Dividir aleatoriamente las imágenes en conjuntos de entrenamiento y validación
random.shuffle(train_image_files)
num_validation_samples = int(0.2 * len(train_image_files))
train_image_files = train_image_files[num_validation_samples:]
validation_image_files = train_image_files[:num_validation_samples]

# Imprimir el número de imágenes cargadas
print("Número de imágenes de entrenamiento:", len(train_image_files))
print("Número de imágenes de validación:", len(validation_image_files))

# Función para cargar las imágenes y las etiquetas manualmente
def load_images_and_labels(image_files):
    images = []
    labels = []
    for file_path in image_files:
        # Leer la imagen
        image = cv2.imread(file_path)
        # Redimensionar la imagen si es necesario
        image = cv2.resize(image, (img_height, img_width))
        # Normalizar los valores de píxeles
        image = image / 255.0
        # Agregar la imagen a la lista de imágenes
        images.append(image)
        # Obtener la etiqueta de la imagen del nombre del directorio
        label = os.path.basename(os.path.dirname(file_path))
        labels.append(label)
    return np.array(images), np.array(labels)

# Cargar manualmente las imágenes y las etiquetas de entrenamiento
train_images, train_labels = load_images_and_labels(train_image_files)

# Convertir las etiquetas a formato one-hot
from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
train_labels_one_hot = label_binarizer.fit_transform(train_labels)

# Crear un DataFrame con las etiquetas codificadas
train_df = pd.DataFrame(train_images, columns=['image'])
train_df['label'] = train_labels_one_hot.tolist()

# Crear el generador de datos a partir del DataFrame
train_data_gen = ImageDataGenerator(rescale=1./255)
train_data = train_data_gen.flow(
    train_df['image'], train_df['label'],
    target_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=True
)

# Cargar los datos de validación de forma similar
validation_images, validation_labels = load_images_and_labels(validation_image_files)
validation_labels_one_hot = label_binarizer.transform(validation_labels)
validation_df = pd.DataFrame(validation_images, columns=['image'])
validation_df['label'] = validation_labels_one_hot.tolist()
val_data = train_data_gen.flow(
    validation_df['image'], validation_df['label'],
    target_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=False
)

# Crear el modelo
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# Entrenar el modelo
epochs = 50
history = model.fit(
    train_data,
    steps_per_epoch=len(train_df) // batch_size,
    epochs=epochs,
    validation_data=val_data,
    validation_steps=len(validation_df) // batch_size
)

# Guardar el modelo
model.save('model.h5')

# Guardar el diccionario de 
# clases
np.save('class_indices.npy', train_data.class_indices)

# Graficar la precisión de entrenamiento y validación
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy Over Epochs')

plt.show()
