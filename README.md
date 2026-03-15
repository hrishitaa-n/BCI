# Brain-Computer Interface (BCI) Motor Imagery Classification

This project implements a **Brain-Computer Interface (BCI) pipeline** that predicts imagined hand movements from EEG brain signals using machine learning.

The system processes EEG signals recorded during **motor imagery tasks** and classifies whether the subject imagined moving their **left hand or right hand**.

---

# Project Overview

Brain–Computer Interfaces allow computers to interpret brain activity and convert it into commands.

In this project, EEG signals are analyzed to detect **motor imagery**, where a subject imagines moving a body part without physically moving.

Example prediction:

EEG Signal → Model → Predicted Movement

EEG Data → Machine Learning Model → LEFT HAND

This approach is commonly used in applications such as:

- Assistive technology for paralysis patients
- Prosthetic limb control
- Neurorehabilitation
- Human–computer interaction systems

---

# Dataset

This project uses the **BCI Competition IV Dataset 2a**, which contains EEG recordings from subjects performing motor imagery tasks.

Each subject imagines one of four movements:

- Left Hand
- Right Hand
- Feet
- Tongue

For simplicity, this project focuses on **binary classification**:

Left Hand vs Right Hand

Dataset source:

https://www.bbci.de/competition/iv/

After downloading the dataset, place the files inside a `data/` directory.

Example:

data/A01T.gdf

---

# Pipeline Overview

The complete system follows this processing pipeline:

EEG Dataset
↓
Bandpass Filtering (8–30 Hz)
↓
Event Extraction
↓
Epoch Extraction (Motor Imagery Trials)
↓
Common Spatial Pattern (CSP) Feature Extraction
↓
Support Vector Machine (SVM) Classifier
↓
Cross-Validation Evaluation
↓
Predicted Motor Imagery Class

---

# Step-by-Step Method

## 1. EEG Data Loading

EEG recordings are loaded from `.gdf` files using the **MNE Python library**.

Each recording contains:

- multi-channel EEG signals
- timestamps
- event markers indicating motor imagery tasks

---

## 2. Signal Filtering

A **bandpass filter (8–30 Hz)** is applied to remove noise and retain the frequency range associated with motor imagery brain activity.

This range includes:

- Mu rhythms
- Beta rhythms

These rhythms change when a person imagines movement.

---

## 3. Event Extraction

Event markers indicate when the subject imagines a movement.

Event codes in the dataset:

| Code | Meaning    |
| ---- | ---------- |
| 769  | Left Hand  |
| 770  | Right Hand |
| 771  | Feet       |
| 772  | Tongue     |

Only **left and right hand trials** are used for classification.

---

## 4. Epoch Extraction

The EEG recording is a continuous signal.

To prepare data for machine learning, the signal is divided into **epochs**, which are short segments of EEG around each motor imagery event.

Each epoch represents **one trial of imagined movement**.

Example:

Trial 1 → Left Hand
Trial 2 → Right Hand
Trial 3 → Left Hand

These trials become training samples for the classifier.

---

## 5. Feature Extraction using CSP

EEG signals contain many channels and time samples, making them difficult for machine learning models to process directly.

To reduce dimensionality, the project uses **Common Spatial Pattern (CSP)**.

CSP finds spatial patterns that maximize the difference between two classes.

For motor imagery tasks:

- Right-hand imagination activates the **left motor cortex**
- Left-hand imagination activates the **right motor cortex**

CSP learns spatial filters that highlight these differences and converts each EEG epoch into a small set of features.

In this project, **four CSP features** are extracted per trial.

---

## 6. Classification using SVM

The extracted CSP features are used to train a **Support Vector Machine (SVM)** classifier.

SVM finds a decision boundary separating feature patterns corresponding to:

Left Hand
Right Hand

Once trained, the model can predict the imagined movement from new EEG data.

---

## 7. Model Evaluation

The model is evaluated using **5-fold cross-validation**.

Process:

1. The dataset is divided into 5 equal parts
2. The model trains on 4 parts
3. The remaining part is used for testing
4. This process repeats five times
5. The average accuracy is reported

Cross-validation provides a more reliable estimate of model performance.

---

# Results

For Subject A01, the model achieved approximately:

Cross-Validation Accuracy ≈ **74.7%**

This means the system correctly predicts the imagined movement in roughly **3 out of 4 trials**.

Typical accuracy ranges for motor imagery EEG classification are between **65% and 80%**, so this result is consistent with expected performance.

---

# Technologies Used

Python
MNE (EEG processing)
NumPy
Scikit-learn
Jupyter Notebook

---

# Project Structure

BCI/
├── eeg_analysis.ipynb
├── dataset-2a.pdf
├── README.md
├── .gitignore

Large dataset files are excluded from the repository using `.gitignore`.

---

# Future Improvements

Possible extensions for this project include:

- Evaluating the pipeline across multiple subjects
- Visualizing CSP spatial patterns
- Testing additional machine learning models
- Building a real-time BCI system

---

# Key Learning Outcomes

This project demonstrates how to build a **complete machine learning pipeline for EEG signal classification**, including:

- signal preprocessing
- feature extraction
- machine learning classification
- model evaluation

It illustrates how brain signals can be translated into computer-interpretable commands using artificial intelligence.

---

# Author

BCI Motor Imagery Classification Project
