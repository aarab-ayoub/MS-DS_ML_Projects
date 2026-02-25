# Deep Learning Project

This repository contains two independent components from the Deep Learning module of an M.Sc. Data Science program.

---

## Structure

```
Deeplearning_project/
├── ocr-from-scratch/      # 🏆 Main project — OCR system built from scratch
│   ├── 01_eda.ipynb
│   ├── 02_training_mps.ipynb
│   ├── results/
│   └── api/               # FastAPI deployment
│
├── course-tps/            # 📚 Course practical work (TP01 → TP11)
│   ├── TP01_introduction_numpy_matplotlib.ipynb
│   ├── TP02_supervised_learning.ipynb
│   ├── TP03_more_ml_models.ipynb
│   ├── TP04_perceptron_spam_detection.ipynb
│   ├── TP05_ann_gradient_descent.ipynb
│   └── TP11_cnn_image_recognition_cifar10.ipynb
│
└── ocr_data/              # Dataset (images + labels.csv)
    └── dataset/
```

---

## OCR from Scratch

> Built by **Aarab Ayoub** & **Merghad Abdelhamid**

A full OCR pipeline built without any pre-trained model:
- **CNN + Transformer** architecture
- **CTC loss** for sequence alignment
- Trained on Apple M4 with **MPS acceleration**
- Deployable as a **REST API** (FastAPI + Docker)

→ [Read more](ocr-from-scratch/README.md) | [Run the API](ocr-from-scratch/api/README.md)

---

## Course TPs

Practical work covering NumPy, scikit-learn, Perceptron, ANN, and CNN on CIFAR-10.

→ [Read more](course-tps/README.md)

---

## Setup

```bash
# Create and activate the virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies (for notebooks)
pip install torch torchvision numpy pandas matplotlib seaborn Pillow tqdm jupyter

# Or install API dependencies
pip install -r ocr-from-scratch/api/requirements.txt
```
