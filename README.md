# SpeechSense

**Speech Emotion Recognition Using Machine Learning**

SpeechSense is a machine learning project that recognizes emotions from speech recordings. The project compares classical machine learning approaches with a convolutional neural network (CNN) using acoustic features extracted from speech.

The goal is to investigate how different feature representations and model architectures perform for speech emotion recognition and eventually build an application capable of classifying emotions from uploaded or recorded speech.

---

## Project Overview

Speech emotion recognition is the task of identifying the emotional state expressed in a person's speech.

In this project, we use the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset and investigate two main approaches:

1. **Classical Machine Learning**
   - MFCC features
   - Logistic Regression
   - Support Vector Machine (SVM)

2. **Deep Learning**
   - Log-Mel Spectrograms
   - Convolutional Neural Network (CNN)

The models are evaluated using a speaker-independent train/validation/test split.

---

## Dataset

### RAVDESS

The project currently uses the speech portion of the RAVDESS dataset.

Dataset characteristics:

- **1,440 speech recordings**
- **24 actors**
- **8 emotion classes**
- Audio-only speech recordings
- Original sample rate: **48 kHz**

### Emotion Classes

| Emotion | Number of Samples |
|---|---:|
| Angry | 192 |
| Calm | 192 |
| Disgust | 192 |
| Fearful | 192 |
| Happy | 192 |
| Neutral | 96 |
| Sad | 192 |
| Surprised | 192 |
| **Total** | **1,440** |

The neutral class contains fewer recordings than the other classes, so metrics such as macro F1-score are also considered instead of relying only on accuracy.

---

## Project Pipeline

```text
RAVDESS Speech Audio
        │
        ▼
   Audio Loading
        │
        ▼
   Preprocessing
   ├── Resample to 16 kHz
   ├── Convert to mono
   └── Normalize amplitude
        │
        ├───────────────────────┐
        ▼                       ▼
      MFCC                 Log-Mel Spectrogram
        │                       │
        ▼                       ▼
Logistic Regression            CNN
        │                       │
        ▼                       ▼
      SVM                   Prediction
        │                       │
        └───────────┬───────────┘
                    ▼
              Model Evaluation
                    │
                    ▼
       Accuracy / Precision / Recall
                    │
                    ▼
              Error Analysis

SpeechSense/
│
├── data/
│   ├── raw/
│   │   ├── Actor_01/
│   │   ├── Actor_02/
│   │   └── ...
│   │
│   └── processed/
│       ├── mfcc.csv
│       ├── mel_spectrograms.npy
│       ├── mel_labels.csv
│       └── model checkpoints
│
├── notebooks/
│
├── src/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── audio.py
│   │
│   ├── features/
│   │   ├── mfcc.py
│   │   └── mel.py
│   │
│   ├── models/
│   │   ├── logistic_regression.py
│   │   ├── svm.py
│   │   └── cnn.py
│   │
│   ├── data/
│   │   └── mel_dataset.py
│   │
│   ├── dataset.py
│   ├── explore.py
│   ├── visualize.py
│   ├── build_features.py
│   ├── build_mel_dataset.py
│   ├── split_data.py
│   ├── train_cnn.py
│   └── evaluate_cnn.py
│
├── experiments/
│
├── app/
│
├── pyproject.toml
├── uv.lock
└── README.md