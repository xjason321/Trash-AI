from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os
import random

def scale_images():
    height = 255
    width = 255
    
    for folder in ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']:
        num_files = len(os.listdir(f"train_data/{folder}")) + 1
        train_number = 1
        test_number = 1
        
        print(f"Folder {folder} has {num_files} images.")
        for i in range(1, num_files):
            path = f'train_data/{folder}/{folder}{str(i)}.jpg'
            im = Image.open(path)
            chance = random.randint(1,20)
            if chance >= 18:
                im.close()
                os.replace(path, f'test_data/{folder}/{folder}{str(test_number)}.jpg')
                test_number += 1
            else:
                im = im.save(f'train_data/{folder}/{folder}{str(train_number)}.jpg')
                train_number += 1
        num_files = len(os.listdir(f"train_data/{folder}")) + 1
        print(f"Folder {folder} has {num_files} images.")
            
def pre_process(path):
    im = Image.open(path)
    pxs = im.load()
    im_list = []
    for i in range(255):
        im_list.append([])
        for j in range(255):
            im_list[i].append(pxs[i, j])
    return (im_list[0])
    
x_train = []
y_train = []

for folder in ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']:
        num_files = len(os.listdir(f"train_data/{folder}")) + 1

        print(f"Folder {folder} has {num_files} images.")
        for i in range(1, num_files):
            path = f'train_data/{folder}/{folder}{str(i)}.jpg'
            x_train.append(pre_process(path))
            y_train.append(folder)
            
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(255, 255, 3)))
#model.add(layers.Conv2D(255, (3, 3), activation='relu', input_shape=(255, 255, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))


model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(256, activation="relu"))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(12, activation="softmax"))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(x_train, y_train, epochs=20)

model.save('handwritten.model')