import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.optimizers import Adam

# from tensorflow.keras import mixed_precision
# mixed_precision.set_global_policy('mixed_float16')

data_path = './data'
class_names = sorted(os.listdir(data_path))

# GPU configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"Using GPU: {gpus[0]}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected.")

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
        if img_name.lower().endswith(('.jpg', '.jpeg')):
            img_path = os.path.join(class_path, img_name)
            image_paths.append(img_path)
            labels.append(class_index)
        else:
            print("Wrong image format:", img_name)
    class_index += 1

image_paths = tf.constant(image_paths)
labels = tf.constant(labels)
dataset_size = len(image_paths)

def preprocess_data(img_path, label):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = preprocess_input(img)
    return img, label

dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

dataset = dataset.shuffle(buffer_size=dataset_size, reshuffle_each_iteration=False)

dataset = dataset.map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)

split = int(dataset_size * 0.9)
train_dataset = dataset.take(split)
test_dataset = dataset.skip(split)

batch_size = 32
train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

for img_batch, label_batch in train_dataset.take(1):
    for i in range(batch_size):
        img = img_batch[i].numpy()
        label = label_batch[i].numpy()
        plt.imshow(img)
        plt.title(label_map[label])
        plt.show()

for img_batch, label_batch in test_dataset.take(1):
    for i in range(batch_size):
        img = img_batch[i].numpy()
        label = label_batch[i].numpy()
        plt.imshow(img)
        plt.title(label_map[label])
        plt.show()


model_path = 'resnet18_epoch_20.keras'
model = load_model(model_path)

for layer in model.layers:
    layer.trainable = True

optimizer = Adam(learning_rate=1e-4)
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    verbose=1,
    mode='min',
    min_lr=1e-6
)

checkpoint = ModelCheckpoint('resnet18_epoch_{epoch:02d}.keras', save_freq='epoch')

model.fit(
    train_dataset,
    validation_data=test_dataset,
    initial_epoch=20,
    epochs=50,
    callbacks=[checkpoint, reduce_lr]
)

model.save('trained_resnet18.keras')
