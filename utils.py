import skimage
import skimage.io
import pandas as pd
import numpy as np
import cv2
import os

def get_batch_data(images_path, images_label, batch_size=32):
    if not os.path.exists(images_path):
        raise ValueError('images_path is not exist.')

    data = pd.read_csv(images_label)
    target_num_map = {0.1: 0, 0.2: 1, 0.5: 2, 1: 3, 2: 4, 5: 5, 10: 6, 50: 7, 100: 8}
    data[' label'] = data[' label'].apply(lambda x: target_num_map[x])
    dict = data.set_index('name').T.to_dict('list')

    images = []
    labels = []
    count = 0
    indices = np.random.choice(39620, batch_size)
    for image_file in os.listdir(images_path):
        count += 1
        if count in indices:
            image = cv2.imread(images_path+'/'+image_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (500, 250), interpolation=cv2.INTER_LINEAR) #959, 472
            # Assume the name of each image is imagexxx_label.jpg
            label = dict[image_file][0]
            images.append(image)
            labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels
