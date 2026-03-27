# 🧠 Brain-Computer Interface (BCI) Motor Imagery Classification

## 📌 Overview

This project implements a **Brain-Computer Interface (BCI)** pipeline to
classify imagined motor movements (**left vs right hand**) from EEG
signals using machine learning.

The system processes EEG recordings from motor imagery tasks and
predicts user intent without physical movement.

---

## 🚀 Applications

- Assistive technology for paralysis patients\
- Prosthetic limb control\
- Neurorehabilitation\
- Human--computer interaction

---

## 📂 Dataset

- **BCI Competition IV Dataset 2a**
- EEG recordings from multiple subjects
- Classes:
  - Left Hand (769)
  - Right Hand (770)

🔗 https://www.bbci.de/competition/iv/

---

## ⚙️ Pipeline

    Raw EEG → Bandpass Filter (8–30 Hz) → Event Extraction → Epoching → CSP → StandardScaler → SVM → Accuracy

### 🔹 Machine Learning Pipeline

```python
Pipeline([
    ('csp', CSP(n_components=4)),
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf'))
])
```

---

## 🧪 Methodology

### 1. EEG Data Loading

- Loaded from `.gdf` files using **MNE**

### 2. Signal Filtering

- Bandpass filter: **8--30 Hz**
- Captures **mu and beta rhythms**

### 3. Event Extraction

Code Meaning

---

769 Left Hand
770 Right Hand

### 4. Epoching

- Time window: **1s--4s**
- Each epoch = one trial

### 5. Feature Extraction & Classification

- CSP (spatial filtering)
- StandardScaler (normalization)
- SVM (RBF kernel)

---

## 📊 Results

### 🔹 Accuracy Across Subjects

- **Average Accuracy:** 0.72\
- **Best Subject (A08T):** 0.96\
- **Worst Subject (A05T):** 0.53

### 🔹 Subject-wise Performance

Subject Accuracy Performance

---

A01T 0.76 Good
A02T 0.61 Moderate
A03T 0.92 Excellent
A04T 0.66 Moderate
A05T 0.53 Poor
A06T 0.65 Moderate
A07T 0.76 Good
A08T 0.96 Excellent
A09T 0.67 Moderate

---

## 🧠 Key Insights

- EEG classification is **highly subject-dependent**
- CSP + SVM performs well but is sensitive to:
  - Noise
  - Subject variability
- Some subjects show strong separability, others do not

---

## 📉 Confusion Matrix Insight (Example: A09T)

- Model shows bias toward predicting **Right Hand**
- Indicates imbalance or feature limitation

---

## 🛠 Installation

```bash
pip install mne scikit-learn numpy pandas matplotlib
```

---

## ▶️ Usage

1.  Load EEG data (.gdf files)
2.  Apply bandpass filtering
3.  Extract events
4.  Create epochs
5.  Train ML pipeline
6.  Evaluate model

---

## 🚀 Future Improvements

- Deep learning models (**EEGNet, CNNs**)
- Hyperparameter tuning (**GridSearchCV**)
- Riemannian geometry approaches
- Real-time BCI system

---

## 📁 Suggested Project Structure

    ├── data/
    ├── notebooks/
    ├── src/
    │   ├── preprocessing.py
    │   ├── feature_extraction.py
    │   ├── model.py
    ├── results/
    │   ├── plots/
    ├── README.md

---

## 📌 Notes

- Power BI dashboard not included due to size constraints
- Results may vary due to EEG variability

---

## ⭐ Acknowledgments

- BCI Competition IV Dataset
- MNE Python Library
- Scikit-learn
