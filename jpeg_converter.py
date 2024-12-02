from PIL import Image
import os

data_path = './test'

def convert_images_to_jpeg(data_path):
    for root, _, files in os.walk(data_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.save(file_path, 'JPEG')
            except Exception as e:
                print(f"Removing invalid image: {file_path}")
                os.remove(file_path)

convert_images_to_jpeg(data_path)