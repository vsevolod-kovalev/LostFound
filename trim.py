# script to ensure that the number of images in each class directory within ./data/
# does not exceed a maximum size (MAX_CLASS_SIZE).

import os
import random

MAX_CLASS_SIZE = 4000

data_path = "./data/"
classes = os.listdir(data_path)
for class_name in classes:
    if not os.path.isdir(data_path + class_name):
        continue
    image_paths = os.listdir(data_path + class_name)
    num_images = len(image_paths)
    num_images_to_remove = num_images - MAX_CLASS_SIZE
    if num_images_to_remove > 0:
        image_ids_to_remove = random.sample(range(0, num_images), num_images_to_remove)
        for id_to_remove in image_ids_to_remove:
            image_to_remove = image_paths[id_to_remove]
            os.remove(data_path + class_name + '/' + image_to_remove)


