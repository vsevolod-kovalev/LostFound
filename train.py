import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

data_path = './data'
class_names = sorted(os.listdir(data_path))

label_map = {}

image_paths = []
labels = []
class_index = 0
for class_name in class_names:
    class_path = os.path.join(data_path, class_name)
    if not os.path.isdir(class_path):
        continue
    print(class_name, class_index)
    label_map[class_index] = class_name
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        image_paths.append(img_path)
        labels.append(class_index)
    class_index += 1

image_paths = tf.constant(image_paths)
labels = tf.constant(labels)

dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

def preprocess_data(img_path, label):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = img / 255.0
    return img, label

dataset = dataset.shuffle(buffer_size=len(image_paths))
dataset = dataset.map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)

for img, label in dataset.take(5):
    plt.imshow(img.numpy())
    plt.title(label_map[label.numpy()])
    plt.show()

split = int(len(image_paths) * 0.8)
train_dataset =  dataset.take(split)
test_dataset = dataset.skip(split)
print(len(train_dataset), len(test_dataset))

batch_size = 32
train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

model = tf.keras.Sequential([
    tf.keras.Input(shape=(224, 224, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(class_index, activation='softmax')
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=15
)
model.save('trained_shallow.keras')


