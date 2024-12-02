import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet import preprocess_input
from constants import class_map

model = tf.keras.models.load_model('trained_resnet18.keras')
test_path = './test/'

def preprocess_data(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = preprocess_input(img)
    return img

overall_total = 0
overall_correct = 0
class_stats = {}
for folder in sorted(os.listdir(test_path)):
    folder_path = os.path.join(test_path, folder)
    if not os.path.isdir(folder_path):
        continue
    
    actual_class_idx = int(folder)
    actual_class_name = class_map[actual_class_idx]
    
    total_images = 0
    correct_predictions = 0
    images_data = []
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        total_images += 1
        overall_total += 1
        
        test_img = preprocess_data(image_path)
        test_img = np.expand_dims(test_img, axis=0)
        
        prediction_probs = model.predict(test_img)
        predicted_class_idx = np.argmax(prediction_probs)
        predicted_class_name = class_map[predicted_class_idx]
        
        is_correct = predicted_class_idx == actual_class_idx
        if is_correct:
            correct_predictions += 1
            overall_correct += 1
        
        img = tf.io.decode_jpeg(tf.io.read_file(image_path))
        images_data.append((img, actual_class_name, predicted_class_name, is_correct))
    
    class_stats[actual_class_name] = {
        "total_images": total_images,
        "correct_predictions": correct_predictions,
        "incorrect_predictions": total_images - correct_predictions,
        "accuracy": correct_predictions / total_images if total_images > 0 else 0,
        "images_data": images_data[:5],
    }

print("\nDetailed Information for Each Class:")
for class_name, stats in class_stats.items():
    print(f"\nClass: {class_name}")
    print(f"  Total images: {stats['total_images']}")
    print(f"  Correct predictions: {stats['correct_predictions']}")
    print(f"  Incorrect predictions: {stats['incorrect_predictions']}")
    print(f"  Accuracy: {stats['accuracy'] * 100:.2f}%")
    
    for img, actual, predicted, correct in stats['images_data']:
        plt.imshow(img)
        title = f"Actual: {actual}, Predicted: {predicted}"
        title += " (Correct)" if correct else " (Incorrect)"
        plt.title(title)
        plt.axis('off')
        plt.show()

overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0
print(f"\nOverall accuracy: {overall_accuracy * 100:.2f}%")
