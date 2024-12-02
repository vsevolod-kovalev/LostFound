
import os

EXPECTED = 4000

data_path = "./data/"
classes = os.listdir(data_path)
size_f = False
for class_name in classes:
    if not os.path.isdir(data_path + class_name):
        continue
    image_paths = os.listdir(data_path + class_name)
    num_images = len(image_paths)
    if num_images != EXPECTED:
        size_f = True
        print("EXPECTED: 4000!")
    print(class_name, num_images)
print(f"\nAll: {EXPECTED}" if not size_f else "Some classes have unexpected size.")

