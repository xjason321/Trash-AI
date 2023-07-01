from PIL import Image
import tensorflow as tf
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
    print(im_list[0])
    
pre_process('train_data/battery/battery1.jpg')

model = tf.keras.Sequential()
