
# Neural Network Hardware Accelerator (Simulation)

## 📌 Overview
This project demonstrates how a **hardware accelerator** can speed up neural network inference.
We simulate a hardware accelerator using **NumPy + Numba (JIT)** on the **MNIST** dataset.

The flow:
1) Train a simple NN on MNIST (TensorFlow/Keras).
2) Measure CPU baseline inference time.
3) Simulate an accelerator by replacing matrix multiplication with a JIT-compiled version.
4) Compare **speed** and **accuracy**, and plot the results.

---

## 🚀 Steps (Scripts)
- `01_train_baseline.py` — trains the model (~97% test accuracy) and saves it.
- `02_cpu_baseline.py` — measures CPU-only inference time + accuracy on 1000 test images.
- `03_accelerator_sim.py` — runs inference using the simulated accelerator (Numba) and reports time + accuracy.
- `04_compare_plots.py` — generates bar charts comparing CPU vs Accelerator (saved in `images/`).

All artifacts (saved model, results) are stored in `artifacts/`.

---

## 📊 Example Results (you will generate your own)
| Metric            | CPU Baseline | Accelerator |
|-------------------|--------------|-------------|
| Accuracy (%)      | ~97          | ~97         |
| Avg Time/Image ms | ~0.8         | ~0.4        |

✅ **~2x faster inference while maintaining accuracy**

---

## 🛠 Tech Stack
- Python 3.9+
- TensorFlow / Keras
- NumPy, Numba
- Matplotlib (for graphs)

---

## ▶️ How to Run

### 1) Create & activate venv (recommended)
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Train the model
```bash
python 01_train_baseline.py
```

### 4) CPU baseline (timing + accuracy on 1000 images)
```bash
python 02_cpu_baseline.py
```

### 5) Accelerator simulation (Numba JIT)
```bash
python 03_accelerator_sim.py
```

### 6) Plots (CPU vs Accelerator)
```bash
python 04_compare_plots.py
# Outputs saved to images/cpu_vs_acc_time.png and images/cpu_vs_acc_acc.png
```

---

## 🎯 Future Work
- Implement on FPGA/ASIC for real hardware acceleration.
- Scale to CNN models (e.g., LeNet/ResNet).
- Optimize energy efficiency in addition to speed.

---

## 👤 Author
- Your Name
- LinkedIn: https://linkedin.com/in/your-link
- Email: your-email@example.com
