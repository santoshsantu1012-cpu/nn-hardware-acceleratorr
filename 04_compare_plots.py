
# 04_compare_plots.py
# Generates bar charts comparing CPU vs Accelerator. Reads json results from artifacts/.
import json
import matplotlib.pyplot as plt

with open("artifacts/cpu_results.json") as f:
    cpu = json.load(f)
with open("artifacts/accel_results.json") as f:
    acc = json.load(f)

cpu_time = cpu["cpu_time_per_image_ms"]
acc_time = acc["accel_time_per_image_ms"]
cpu_acc = cpu["cpu_accuracy_pct"]
acc_acc = acc["accel_accuracy_pct"]

# Plot 1: Speed
plt.figure(figsize=(6,4))
plt.bar(["CPU", "Accelerator"], [cpu_time, acc_time])
plt.ylabel("Time per Image (ms)")
plt.title("Inference Speed Comparison")
plt.tight_layout()
plt.savefig("images/cpu_vs_acc_time.png", dpi=160)
plt.close()

# Plot 2: Accuracy
plt.figure(figsize=(6,4))
plt.bar(["CPU", "Accelerator"], [cpu_acc, acc_acc])
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Comparison")
plt.ylim(95, 100)
plt.tight_layout()
plt.savefig("images/cpu_vs_acc_acc.png", dpi=160)
plt.close()

print("Saved plots to images/cpu_vs_acc_time.png and images/cpu_vs_acc_acc.png")
