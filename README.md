# 🎓 MSc — Master's in Data Science Journey

> A two-year archive of coursework, practical labs (TPs), and graded projects completed during my **Master of Science in Data Science**.
>
> Maintained by **Ayoub Aarab**.

This repository collects everything I built and studied throughout the program — from foundational statistics and machine learning to deep learning, probabilistic modeling, IoT, embedded systems, and applied software projects. Each folder is a self-contained module or project, often with its own notebooks, reports, and documentation.

---

## 📚 Table of Contents

- [Repository Structure](#-repository-structure)
- [Modules & Projects](#-modules--projects)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Author](#-author)

---

## 🗂️ Repository Structure

```
MSc/
├── Deeplearning_project/      # Deep Learning module — OCR from scratch + course TPs
├── ml_probabilistic/          # Probabilistic ML — stochastic reliability modeling
├── NLARMA/                    # Time-series forecasting — ARMA vs NLARMA study
├── IOT/                       # Internet of Things — ECG / PTB-XL analysis
├── PetVision/                 # Applied web project — Django + Tailwind app
├── Statistique_pour_bigData/  # Statistics for Big Data — Ridge regression
├── embedded_system/           # Embedded systems — Arduino TPs
├── traitement_des_données/    # Data processing — GPGPU histogram equalization
├── sec/                       # Security / DFIR exercises
├── cours/                     # Lecture code & exercises (regression, trees, etc.)
└── results/                   # Shared figures and summary outputs
```

---

## 🧩 Modules & Projects

### 🧠 Deep Learning — `Deeplearning_project/`
A full **OCR system built from scratch** (no pre-trained models), pairing a **CNN + Transformer** architecture with **CTC loss** for sequence alignment, trained on Apple Silicon (MPS) and deployable as a **FastAPI + Docker** REST API. Alongside the main project, `course-tps/` holds the module's practical work (TP01 → TP11), covering NumPy fundamentals, supervised learning, perceptrons, gradient descent, and CNNs on CIFAR-10.

### 🎲 Probabilistic Machine Learning — `ml_probabilistic/`
**Stochastic Reliability Modeling for H100 GPU Clusters** — a unified pipeline covering DTMC/CTMC reliability analysis, hidden Markov models, MDP control, reinforcement learning (Q-learning, SARSA, TD(0), TD(λ)), POMDP belief updates, and a non-Markovian memory extension. Includes a LaTeX report and reproducible experiment modes.

### 📈 Time-Series Forecasting — `NLARMA/`
A comparative analysis of **ARMA vs. NLARMA** models for time-series forecasting, with optimization code, a written report, and a presentation.

### 📡 Internet of Things — `IOT/`
ECG signal analysis using the **PTB-XL** dataset and collective-intelligence experiments, documented in the accompanying project report.

### 🐾 PetVision — `PetVision/`
An applied **Django** web application styled with **Tailwind CSS** — a full project with models, views, templates, and migrations.

### 📊 Statistics for Big Data — `Statistique_pour_bigData/`
A clean implementation and study of **Ridge Regression** on a real dataset.

### 🔌 Embedded Systems — `embedded_system/`
**Arduino** practical work (TP1–TP4): digital/analog I/O, serial communication, and sensor reads.

### 🖼️ Data Processing — `traitement_des_données/`
Data-processing labs including a **GPGPU histogram-equalization** project and supporting notebooks.

### 🛡️ Security — `sec/`
Digital forensics & incident response (**DFIR**) exercises.

### 📖 Course Code — `cours/`
Lecture-along scripts and exercises: linear/multiple/polynomial regression, decision trees, and 3D visualizations.

---

## 🛠️ Tech Stack

- **Languages:** Python, C/C++ (GPGPU), LaTeX, SQL
- **ML / DL:** PyTorch, scikit-learn, NumPy, pandas, Matplotlib
- **Web:** Django, Tailwind CSS, FastAPI
- **Tooling:** Jupyter, Docker, Arduino
- **Domains:** Deep learning, probabilistic & stochastic modeling, time series, IoT, embedded systems, statistics

---

## 🚀 Getting Started

Each module is self-contained. To explore a project, navigate into its folder and follow its local `README.md` where available:

```bash
git clone <repo-url>
cd MSc/<module>
```

Most Python projects ship a `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 👤 Author

**Ayoub Aarab** — M.Sc. Data Science

> This repository documents an ongoing academic journey; modules and projects are added as the program progresses.
