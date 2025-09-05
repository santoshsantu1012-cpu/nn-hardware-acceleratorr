
# 01_train_baseline.py
# Trains a simple MNIST classifier and saves the model to artifacts/model.keras

import os
import tensorflow as tf
from tensorflow.keras import layers, models, datasets

os.makedirs("artifacts", exist_ok=True)

# Load MNIST
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build simple model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=2)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Save model
model.save("artifacts/model.keras")
print("Saved model to artifacts/model.keras")
