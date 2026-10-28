"""Predict the CIFAR-10 class of a single image using the trained MobileNetV2 model.

Usage:
    python inference.py --image_path OIP.jpg
    python inference.py --image_path photo.png --model_path cifar10_mobilenetv2.keras --top_k 3
"""
import argparse
import os
import sys

import numpy as np
import tensorflow as tf
from PIL import Image

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


def load_image(image_path):
    """Load an image as a (1, 32, 32, 3) float32 array.

    The model resizes to 96x96 and applies MobileNetV2 preprocessing internally,
    so only the CIFAR-10 sized input (raw 0-255 pixels) is prepared here.
    """
    img = Image.open(image_path).convert('RGB').resize((32, 32))
    return np.expand_dims(np.array(img, dtype=np.float32), axis=0)


def predict_image(image_path, model_path='cifar10_mobilenetv2.keras', top_k=3):
    if not os.path.exists(model_path):
        sys.exit(f"Model file '{model_path}' not found. Run train.ipynb first.")
    if not os.path.exists(image_path):
        sys.exit(f"Image '{image_path}' not found.")

    model = tf.keras.models.load_model(model_path)
    probs = model.predict(load_image(image_path), verbose=0)[0]

    best = int(np.argmax(probs))
    print(f"Image: {image_path}")
    print(f"Predicted Class: {CLASS_NAMES[best]}")
    print(f"Confidence: {probs[best] * 100:.2f}%")

    if top_k > 1:
        print(f"\nTop {top_k} predictions:")
        for i in np.argsort(probs)[::-1][:top_k]:
            print(f"  {CLASS_NAMES[i]:<11} {probs[i] * 100:6.2f}%")
    return CLASS_NAMES[best], float(probs[best])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-10 image classifier (MobileNetV2)")
    parser.add_argument('--image_path', default='OIP.jpg', help='Path to the image to classify')
    parser.add_argument('--model_path', default='cifar10_mobilenetv2.keras', help='Path to the saved model')
    parser.add_argument('--top_k', type=int, default=3, help='Number of top predictions to show')
    args = parser.parse_args()
    predict_image(args.image_path, args.model_path, args.top_k)
