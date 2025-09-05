
# 02_cpu_baseline.py
# Loads the saved model and measures CPU inference time + accuracy on 1000 test images.
import json, time
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, models

# Load data
(_, _), (x_test, y_test) = datasets.mnist.load_data()
x_test = x_test / 255.0

# Load model
model = models.load_model("artifacts/model.keras")

# Select subset
N = 1000
sample_images = x_test[:N]
sample_labels = y_test[:N]

# Timing
start_time = time.time()
preds = model.predict(sample_images, verbose=0)
end_time = time.time()

# Metrics
total_time = end_time - start_time
time_per_image_ms = (total_time / N) * 1000.0
pred_labels = np.argmax(preds, axis=1)
accuracy = (pred_labels == sample_labels).mean() * 100.0

print(f"CPU Baseline — Total time for {N}: {total_time:.4f}s")
print(f"CPU Baseline — Avg time/image: {time_per_image_ms:.4f} ms")
print(f"CPU Baseline — Accuracy: {accuracy:.2f}%")

# Save results
results = {
    "cpu_total_time_s": float(total_time),
    "cpu_time_per_image_ms": float(time_per_image_ms),
    "cpu_accuracy_pct": float(accuracy),
    "N": int(N)
}

with open("artifacts/cpu_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Saved CPU results to artifacts/cpu_results.json")
