# 🛌 Sleep Disorder Detection using Audio-Based Machine Learning

## 📌 Overview

This project focuses on detecting multiple sleep disorders — particularly **Obstructive Sleep Apnea (OSA)** — using **audio signals and machine learning techniques**.

The system processes sleep audio recordings, extracts meaningful features, and applies both **classical machine learning** and **deep learning models** to classify events such as:
- Obstructive Apnea
- Hypopnea
- Snore
- Desaturation

An optimized **ensemble model** is used to improve detection performance, especially for clinically important events.

---

## 🎯 Objectives

- Detect sleep-related breathing disorders using audio signals  
- Compare classical ML and deep learning approaches  
- Improve performance using ensemble learning  
- Focus on **high recall for Obstructive Apnea** (critical in medical diagnosis)  

---

## 📂 Dataset

This project uses the **APSAA (Audio-Polygraphy Sleep Apnea Analysis) dataset**, which includes:

- Audio recordings (.wav)
- Event annotations (CSV)
- Multiple patients (patient-wise data)

---

## 🔄 Data Processing Pipeline

### 1. Event Segmentation
- Audio split into **10-second clips** based on annotation timestamps
- Each clip corresponds to a labeled event

### 2. Preprocessing
- Resampling to 16 kHz  
- Normalization  
- Padding/cropping to fixed length  

---

## 🎧 Feature Extraction

Extracted features:
- **MFCC (Mel-Frequency Cepstral Coefficients)**
- **Log-Mel Spectrogram**
- **LFCC**
- **CQCC**

Two feature formats were used:

### 📊 Tabular Features
- Mean + standard deviation of features
- Used for classical ML models

### 🧠 Spectrogram Features
- Shape: **(64, 400, 2)** → (Frequency × Time × Channels)
- Channels = MFCC + Log-Mel
- Used for CNN and CRNN

---

## 📊 Dataset Split

Patient-wise split (to avoid data leakage):

35 patients → ~24,000 events → Train (~16,422) | Validation (~3,569) | Test (~3,566)

---

## 🤖 Models Used

### 🔹 Classical Machine Learning
- Logistic Regression
- Random Forest
- SVM (RBF Kernel)
- **XGBoost (Best classical model)**

### 🔹 Deep Learning
- CNN (Convolutional Neural Network)
- CRNN (CNN + BiLSTM)

---

## 📈 Evaluation Metrics

- Precision
- Recall
- F1-Score

📌 Special focus:
Recall for Obstructive Apnea (to minimize missed detections)

---

## 🚀 Ensemble Model (Final Solution)

A weighted ensemble combining:

- **XGBoost (55%)**
- **CNN (25%)**
- **CRNN (20%)**

### 🔧 Threshold Optimization
- Custom threshold applied for Obstructive Apnea
- Improves detection sensitivity

---

## 📊 Results

### 🔹 Best Individual Model (XGBoost)
- Precision: 0.7784  
- Recall: 0.6813  
- F1-score: 0.7266  

### 🔹 Final Ensemble Model
- Precision: 0.8155  
- Recall: 0.7001  
- F1-score: 0.7534  

📈 Improvement:
- F1-score increased by ~3.7%

---

## 📁 Project Structure

preprocessing/  
feature_extraction/  
splits/  
models/  
ensemble/  
results/  

---

## 🛠️ Technologies Used

- Python
- NumPy, Pandas
- Librosa
- Scikit-learn
- XGBoost
- TensorFlow / Keras
- Matplotlib

---

## 🧠 Key Learnings

- Audio signals can effectively detect sleep disorders  
- Ensemble methods improve reliability  
- Medical applications require recall-focused optimization  

---

## ⚠️ Limitations

- Audio-only system may miss physiological signals  
- Class imbalance affects performance  
- Not a replacement for clinical diagnosis  

---

## 🔮 Future Work

- Add physiological sensors (SpO₂, airflow, thorax)
- Real-time monitoring
- Mobile deployment
- Multi-modal learning  

---

## 📌 Conclusion

Audio-based ML combined with ensemble techniques provides a scalable solution for sleep disorder screening.

---

## 👤 Author

Naman Kumar
