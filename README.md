# Project Logbook  Sem Project
### **Project Title:** AI Based Document Fraud Detection Web System Using OCR and Deep Learning

| Sr.No |                           Contents                                    | Date        |
|-------|-----------------------------------------------------------------------|-------------|
| 1     | Project Group Formation.                                              |   |
| 2     | Project Topic Finalization.                                           |   |
| 3     | Guide suggested to focus on “AI Based Document Fraud Detection Web System Using OCR and Deep Learning” and start with Requirement Gathering, Existing System Study, and discuss project scope and feasibility. |   |
| 5     | Discussion on Software and Hardware Requirements with guide: Finalized tools, technologies, and platform to be used for the project. |   |
| 6     | Abstract – Short Summary/Overview of Project prepared.                |   |
| 7     | Home page design (Basic layout for room listing and search feature).  |   |
| 8     | Dataset Creation on Kaggle by own.                                   |   |
| 9     | ML Model Testing                           |   |
| 10    | Designing and Implementation Progress – 1 / Project Monitoring – 1.   |   |
| 11    | Implementation Progress-2                                             |   |
| 12    | Project Interaction - Demo with Guide (Frontend) |    |
| 13    | Implementation Progress – 3 / Project Monitoring – 2.                 |   |
| 14    |                |   |
| 15    |               |   |
| 16    |               |   |

---
# AI-Based Document Fraud Detection Web System Using Deep Learning

## Overview

The AI-Based Document Fraud Detection Web System is a deep learning-powered web application designed to automatically detect fraudulent and tampered documents. The system analyzes uploaded document images and classifies them as **Genuine** or **Fraudulent** using a Convolutional Neural Network (CNN) model trained on document datasets.

The project aims to reduce manual verification efforts and improve the accuracy, speed, and reliability of document authentication in educational institutions, recruitment agencies, financial organizations, and government sectors.

---

## Features

- Document Image Upload
- Deep Learning-Based Fraud Detection
- Real-Time Prediction
- Genuine/Fake Classification
- Confidence Score Generation
- Responsive Web Interface
- Automated Verification Process
- Easy Deployment and Scalability

---

## Project Architecture

```text
User Uploads Document
          │
          ▼
 Image Preprocessing
          │
          ▼
 Feature Extraction
          │
          ▼
 CNN Deep Learning Model
          │
          ▼
 Fraud Detection Prediction
          │
          ▼
 Display Result on Web Interface
```

---

## Technology Stack

### Frontend
- HTML5
- CSS3
- JavaScrip

### Backend
- Python
- Flask

### Deep Learning
- TensorFlow
- Keras
- OpenCV
- NumPy
- Pandas

### Model Training
- CNN (Convolutional Neural Network)

---

## Dataset

The model is trained using internship certificate and document image datasets containing:

- Genuine Documents
- Fraudulent Documents
- Tampered Certificates
- Modified Document Images

Dataset Structure:

```text
dataset/
│
├── genuine/
│   ├── doc1.jpg
│   ├── doc2.jpg
│   └── ...
│
└── fake/
    ├── doc1.jpg
    ├── doc2.jpg
    └── ...
```

---

## Deep Learning Workflow

### 1. Data Collection
Collect genuine and fraudulent document images.

### 2. Data Preprocessing
- Resize images
- Normalize pixel values
- Convert image format
- Remove noise

### 3. Model Training
Train a CNN model using TensorFlow/Keras.

### 4. Model Evaluation
Evaluate using:
- Accuracy
- Precision
- Recall
- F1 Score

### 5. Deployment
Deploy trained model using Flask Web Framework.

---

## Project Structure

```text
AI-Document-Fraud-Detection/
│
├── app.py
├── train.py
├── predict.py
├── model/
│   └── fraud_detection_model.h5
│
├── dataset/
│   ├── genuine/
│   └── fake/
│
├── static/
│   ├── css/
│   ├── js/
│   └── uploads/
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── requirements.txt
├── README.md
└── notebook.ipynb
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/PLACEHOLDER/AI-Document-Fraud-Detection.git

cd AI-Document-Fraud-Detection
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Model Training

Run:

```bash
python train.py
```

The trained model will be saved as:

```text
fraud_detection_model.h5
```

---

## Run Web Application

```bash
python app.py
```

Open browser:

```text
http://127.0.0.1:5000
```

---

## Prediction Process

1. Upload document image.
2. Image is preprocessed.
3. CNN model extracts features.
4. Model predicts fraud probability.
5. Result displayed on webpage.



## Performance Metrics

| Metric | Value |
|----------|----------|
| Accuracy | 84.1% |
| Precision | - |
| Recall | - |
| F1 Score | 0.8403 |

---

## Applications

- Educational Certificate Verification
- Recruitment Document Validation
- Government Record Authentication
- Banking and Financial Institutions
- Insurance Claim Verification
- Corporate Background Verification


## Team Members:


Pranav Chaudhari

Darshan Shinde

Harsh Mali

Pratik Nikwade

## Authors

**College:** R C Patel Institute of Technology

**Department:** Computer Science and Engineering (Data Science)

**Guide:** Prof. Priyanka D. Lanjewar

**Academic Year:** 2025-2026

---

## License

This project is developed for academic and research purposes only.
