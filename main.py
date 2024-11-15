import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.layers import BatchNormalization
import random
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint

os.environ['TF_DISABLE_MEMORY_OPTIMIZATION'] = '1'


def load_breakhis_data(train_dir, test_dir):
    label_to_index = {'benign': 0, 'malignant': 1}

    def load_images_from_directory(directory):
        images = []
        labels = []
        for label_name in os.listdir(directory):
            label_index = label_to_index[label_name]
            label_dir = os.path.join(directory, label_name)
            for image_file in os.listdir(label_dir):
                image_path = os.path.join(label_dir, image_file)
                image = cv2.imread(image_path)
                image = cv2.resize(image, (700, 460))
                images.append(image)
                labels.append(label_index)
        return np.array(images), np.array(labels)

    x_train, y_train = load_images_from_directory(train_dir)
    x_test, y_test = load_images_from_directory(test_dir)

    return x_train, y_train, x_test, y_test

path_train = "/kaggle/input/breakhis-400x/BreaKHis 400X/train"
path_test =  "/kaggle/input/breakhis-400x/BreaKHis 400X/test"

x_train, y_train, x_test, y_test = load_breakhis_data(path_train, path_test)

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)

from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam

x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.2, random_state=42)

print( x_val.shape)
print(y_val.shape)

from tensorflow.keras.applications.vgg16 import preprocess_input

x_train_preprocessed = preprocess_input(x_train)
x_val_preprocessed = preprocess_input(x_val)

base_model = VGG16(weights='/kaggle/input/vgg16-rares/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, input_shape=(460, 700, 3))

for layer in base_model.layers:
    layer.trainable = False

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(460, 700, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')
])

model.summary()

checkpoint = ModelCheckpoint('best_model_weights.keras',
                             monitor='val_accuracy',
                             save_best_only=True,
                             mode='max')

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=12, validation_data=(x_val, y_val), callbacks=[checkpoint])
x_test_preprocessed = preprocess_input(x_test)

loss, accuracy = v16_model.evaluate(x_test_preprocessed, y_test)
int('test loss, test acc:',loss, accuracy )