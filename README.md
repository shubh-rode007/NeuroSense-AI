# 🧠 NeuroSense AI – Multi-Modal Parkinson’s Detection System

A **production-ready AI healthcare system** designed for early detection of Parkinson’s Disease using multiple modalities:

🎤 Voice | 👃 Smell | 🖐️ Gesture | ✍️ Handwriting

---

## 🚀 Project Overview

NeuroSense AI integrates **four independent AI models** into a unified system to improve early-stage Parkinson’s detection accuracy.

Unlike traditional single-input systems, this solution leverages **multi-modal analysis**, making predictions more reliable and practical for real-world use.

---

## 🎯 Key Highlights

* 🔹 Multi-modal AI system (4 different ML/DL models)
* 🔹 Real-time gesture detection using webcam
* 🔹 Browser-based voice recording & analysis
* 🔹 CNN-based handwriting classification
* 🔹 Interactive smell test with ML prediction
* 🔹 Premium, consistent UI (Glassmorphism design)
* 🔹 Fully integrated full-stack application

---

## 🧩 Features

### 🎤 Voice Analysis

* Records audio directly from browser
* Extracts features using **Librosa**
* Predicts Parkinson symptoms using trained ML model

---

### 👃 Smell Test

* Dynamic question-based smell identification
* Score-based + ML-based prediction
* Displays:

  * Smell score
  * Risk percentage
  * Interpretation level

---

### 🖐️ Gesture Recognition

* Real-time hand tracking using **MediaPipe**
* Live webcam feed processing
* Stable predictions using buffer smoothing

---

### ✍️ Handwriting Analysis

* Upload handwriting samples
* CNN model for classification
* Image preprocessing (grayscale, resize, normalization)

---

## 🧠 System Architecture

```id="ybbj6t"
User Input
   ↓
[ Voice | Smell | Gesture | Handwriting ]
   ↓
Individual AI Models
   ↓
Predictions
   ↓
( Future ) Combined AI Risk Score
```

---

## 🖥️ Tech Stack

### Frontend

* HTML, CSS (Glass UI), JavaScript

### Backend

* Flask (Python)

### Machine Learning / AI

* Scikit-learn
* TensorFlow / Keras
* MediaPipe
* Librosa

---


## 🎥 Demo

> 📌 Demo video- Flow of Complete System 

```id="c7l2ns"
[▶ Watch Demo](https://your-demo-link.com)
```

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```id="v3spjo"
git clone (https://github.com/Pandurang2004/NeuroSense-AI.git )
cd your-repo-name
```

---

### 2️⃣ Install Dependencies

```id="lxr8k1"
pip install -r requirements.txt
```

---

### 3️⃣ Install FFmpeg (IMPORTANT)

Download from:
👉 https://www.gyan.dev/ffmpeg/builds/

Add the **bin folder** to system PATH.

Verify:

```id="ycfz9j"
ffmpeg -version
```

---

### 4️⃣ Run Application

```id="wt7jzq"
python app.py
```

---

## 🌐 Access

* Backend App → http://127.0.0.1:5000
* Dashboard → http://127.0.0.1:8000/dashboard.html

---

## 🎯 Future Improvements

* 📊 Combined AI Risk Dashboard
* 📄 PDF Medical Report Generation
* Build Mobile App 

---

## 👨‍💻 Author

**Pandurang Sidram Bhosale**

* 🎓 CSE Student (2026 Batch)
* 💡 AI/ML & Full Stack Developer

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
