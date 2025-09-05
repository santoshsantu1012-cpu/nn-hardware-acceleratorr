
# 03_accelerator_sim.py
# Simulates an accelerator by using a JIT-compiled matrix multiplication for forward pass.
import json, time
import numpy as np
from numba import jit
import tensorflow as tf
from tensorflow.keras import datasets, models

# JIT-compiled matrix multiplication
@jit(nopython=True, parallel=True, fastmath=True)
def accel_matmul(A, B):
    return A @ B

def accelerator_forward(x, w1, b1, w2, b2):
    x = x.reshape(1, 784)
    h = accel_matmul(x, w1) + b1
    h = np.maximum(h, 0)  # ReLU
    o = accel_matmul(h, w2) + b2
    # Softmax
    o = o - np.max(o)
    exp_o = np.exp(o)
    probs = exp_o / np.sum(exp_o)
    return int(np.argmax(probs))

# Load dataset
(_, _), (x_test, y_test) = datasets.mnist.load_data()
x_test = x_test / 255.0

# Load trained model & extract weights
model = models.load_model("artifacts/model.keras")
# layers: Flatten (0), Dense(1), Dense(2)
w1, b1 = model.layers[1].get_weights()
w2, b2 = model.layers[2].get_weights()

# Ensure arrays are contiguous and of type float64/float32 as needed
w1 = np.ascontiguousarray(w1)
b1 = np.ascontiguousarray(b1)
w2 = np.ascontiguousarray(w2)
b2 = np.ascontiguousarray(b2)

# Warm-up JIT (compile once)
_ = accel_matmul(np.zeros((1, w1.shape[0])), w1)

# Evaluate on subset
N = 1000
sample_images = x_test[:N]
sample_labels = y_test[:N]

start = time.time()
correct = 0
for i in range(N):
    pred = accelerator_forward(sample_images[i], w1, b1, w2, b2)
    if pred == sample_labels[i]:
        correct += 1
end = time.time()

accuracy = (correct / N) * 100.0
time_per_image_ms = ((end - start) / N) * 1000.0

print(f"Accelerator — Avg time/image: {time_per_image_ms:.4f} ms")
print(f"Accelerator — Accuracy: {accuracy:.2f}%")


# Save results
results = {
    "accel_time_per_image_ms": float(time_per_image_ms),
    "accel_accuracy_pct": float(accuracy),
    "N": int(N)
}
with open("artifacts/accel_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Saved Accelerator results to artifacts/accel_results.json")
