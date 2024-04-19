import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

# Deshabilitar el modo ansioso y el modo de depuración para tf.data
# Si deseas probar más adelante, puedes volver a habilitarlos.
# tf.config.run_functions_eagerly(True)
# tf.data.experimental.enable_debug_mode()

# Directorio de datos
train_data_dir = 'Entrenamiento/osvaldo'

# Parámetros
img_width, img_height = 150, 150
epochs = 50
# Reducir el tamaño del lote
batch_size = 12

# Crear el modelo
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Preparar los datos
train_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

# Verificar clases encontradas
print(train_generator.class_indices)

# Entrenar el modelo
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs)

# Guardar el modelo entrenado
model.save('model3.h5')
