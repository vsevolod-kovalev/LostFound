# script to top off the classes with <4000 images with augmented data

import albumentations as A
from PIL import Image
import numpy as np

import os
import random
import uuid

MIN_CLASS_SIZE = 4000

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=45, p=0.5)
])

data_path = "./data/"
classes = os.listdir(data_path)
for class_name in classes:
    class_path = os.path.join(data_path, class_name)
    if not os.path.isdir(class_path):
        continue
    image_paths = os.listdir(class_path)
    num_images = len(image_paths)
    num_images_to_add = MIN_CLASS_SIZE - num_images
    if num_images_to_add > 0:
        ids_to_augment = random.choices(range(num_images), k=num_images_to_add)
        for image_id in ids_to_augment:
            image_filename = image_paths[image_id]
            image_path = os.path.join(class_path, image_filename)
            try:
                image = np.array(Image.open(image_path))
                augmented = transform(image=image)
                augmented_image = augmented['image']
                augmented_image = np.clip(augmented_image, 0, 255).astype(np.uint8)
                augmented_pil_image = Image.fromarray(augmented_image)
                base_name, ext = os.path.splitext(image_filename)
                unique_id = uuid.uuid4().hex[:6]
                new_image_name = f"{base_name}_aug_{unique_id}{ext}"
                new_image_path = os.path.join(class_path, new_image_name)
                augmented_pil_image.save(new_image_path)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        print(f"{class_name}: Augmented {num_images_to_add} images to reach {MIN_CLASS_SIZE}")
    else:
        print(f"{class_name}: Already has {num_images} images, no augmentation needed.")
