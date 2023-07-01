from PIL import Image
import tensorflow as tf
import numpy as np
import os
import random

def cut_down_data():
    for folder in ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']:
        num_files = len(os.listdir(f"train_data/{folder}")) 
        index = 1

        print(f"Folder {folder} has {num_files} images.")
        
        for file in os.listdir(f'train_data/{folder}'):
            path = f'train_data/{folder}/{folder}_{str(index)}.jpg'
            im = Image.open(f'train_data/{folder}/{file}')
            os.remove(f'train_data/{folder}/{file}')
            im = im.save(path)
            index += 1

cut_down_data()