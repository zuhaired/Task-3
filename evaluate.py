"""Evaluate the saved model on the CIFAR-10 test set.

Prints overall accuracy plus per-class precision / recall / F1 and saves
confusion_matrix.png and metrics.txt.

Usage: python evaluate.py [--model_path cifar10_mobilenetv2.keras]
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default='cifar10_mobilenetv2.keras')
args = parser.parse_args()

model = tf.keras.models.load_model(args.model_path)
(_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_true = y_test.flatten()

y_pred = np.argmax(model.predict(x_test, batch_size=128, verbose=1), axis=1)

acc = (y_pred == y_true).mean()
report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
print(f"Test accuracy: {acc * 100:.2f}%\n")
print(report)
with open('metrics.txt', 'w') as f:
    f.write(f"Test accuracy: {acc * 100:.2f}%\n\n{report}")

cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(8, 8))
ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax, xticks_rotation=45, colorbar=False)
ax.set_title('CIFAR-10 Confusion Matrix (test set)')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=120)
print("Saved confusion_matrix.png and metrics.txt")
