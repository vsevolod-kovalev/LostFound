import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

image_net_map = {
    0: "backpack",
    1: "book",
    2: "camera",
    3: "laptop",
    4: "phone",
    5: "running_shoes",
    6: "sunglasses",
    7: "umbrella",
    8: "wallet",
    9: "watch",
    10: "water_bottle"
}

def preprocess_data(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = img / 255.0
    return img
image_path = "wallet.jpeg"
test_img = preprocess_data(image_path)
plt.imshow(test_img)

test_img = np.array([test_img])

model = tf.keras.models.load_model('trained_shallow.keras')
prediction = np.argmax(model.predict(test_img))

plt.title(f"Prediction: {image_net_map[prediction]}")
plt.show()