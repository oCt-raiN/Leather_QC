# -*- coding: utf-8 -*-
"""leather_defect.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MadYNgQyd3RxiXFjQrssAr-pwyDPWhwh
"""

# !git clone https://github.com/oCt-raiN/Leather_QC.git

import os

!unzip /content/drive/MyDrive/leather_dataset/leather_dataset.zip

# one time for filename change

# dir = "/content/Leather_Defect_Classification/"
# for i in os.scandir(dir):
#   for j in os.scandir(i):
#     if j.is_file():
#       os.rename(j.path,j.path.replace(" ","_"))

# !zip -r leather_dataset.zip /content/Leather_Defect_Classification/

# !mv /content/leather_dataset.zip /content/drive/MyDrive/leather_dataset/

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/content/content/Leather_Defect_Classification'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import numpy as np
import pandas as pd
import os
import tensorflow as tf

train_dir = "/content/content/Leather_Defect_Classification"

train_ds = tf.keras.utils.image_dataset_from_directory(
  train_dir,
  validation_split=0.2,
   color_mode='grayscale',
  subset="training",
  seed=123,
  image_size=(227, 227),
  batch_size=64)

val_ds = tf.keras.utils.image_dataset_from_directory(
  train_dir,
  validation_split=0.2,
  subset="validation",
    color_mode='grayscale',
  seed=123,
  image_size=(227, 227),
  batch_size=64)

train_ds.get_single_element

class_names = train_ds.class_names
print(class_names)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    print(images[i].shape)
    plt.imshow(images[i].numpy().astype("uint8"))

    plt.title(class_names[labels[i]])
    plt.axis("off")

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

normalization_layer = tf.keras.layers.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = 6
from keras import applications

# model = tf.keras.Sequential([
#   tf.keras.layers.Rescaling(1./255),
#   tf.keras.layers.Conv2D(32, 3, activation='relu'),
#   tf.keras.layers.MaxPooling2D(),
#   tf.keras.layers.Conv2D(32, 3, activation='relu'),
#   tf.keras.layers.MaxPooling2D(),
#   tf.keras.layers.Conv2D(32, 3, activation='relu'),
#   tf.keras.layers.MaxPooling2D(),
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dense(num_classes)
# ])
img_height=227
img_width=227
# base_model = applications.resnet50.ResNet50(weights= None, include_top=False,
#input_shape= (img_height,img_width,1))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.applications.vgg16 import VGG16

# define input size
input_shape = (227, 227, 1)

# define number of output classes
num_classes = 6

# load pre-trained VGG16 model with ImageNet weights
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(227, 227, 3))

# create the model
model = Sequential()

# add convolutional layers to replicate grayscale image across three channels
model.add(Conv2D(3, (3, 3), padding='same', input_shape=input_shape))

# add VGG16 layers
for layer in vgg16.layers:
    model.add(layer)

# freeze pre-trained layers
for layer in model.layers:
    layer.trainable = False

# add new classification layers
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dense(4096, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# print model summary
model.summary()

from keras.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(filepath='weights.{epoch:02d}-{val_loss:.2f}.h5',
                                      monitor='val_loss',
                                      verbose=1,
                                      save_best_only=True,
                                      save_weights_only=True,
                                      mode='auto',
                                      period=1)

# model.fit(x_train, y_train,
#           epochs=10,
#           batch_size=32,
#           validation_data=(x_val, y_val),
#           callbacks=[checkpoint_callback])

# model=resnet50()
model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'],
)

history=model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=5,
    callbacks=[checkpoint_callback]
)

history=history.history

plt.plot(history['loss'], label='Training Loss')

# Plot validation loss if available
if 'val_loss' in history:
    plt.plot(history['val_loss'], label='Validation Loss')

# Add plot labels and legend
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.ylim(0, 10)


# Show plot
plt.show()

model.save('/content/drive/MyDrive/leather_dataset/leather_defect_model', save_format='tf')