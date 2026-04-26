<h1 align="center">🧠 NeuroHub — AI Face Verification Microservice</h1>

<p align="center">
  A biometric face verification service powered by a <strong>Siamese Neural Network</strong> — built with TensorFlow, Flask, and Python.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Flask-REST%20API-000000?style=for-the-badge&logo=flask&logoColor=white"/>
  <img src="https://img.shields.io/badge/Siamese%20Network-Face%20ID-blueviolet?style=for-the-badge"/>
</p>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [How It Works](#-how-it-works)
- [Project Structure](#-project-structure)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Model Setup](#-model-setup)
- [Running the Server](#-running-the-server)
- [API Reference](#-api-reference)
- [Thresholds Explained](#-thresholds-explained)
- [Integration with Medify](#-integration-with-medify)
- [License](#-license)

---

## 🌟 Overview

**NeuroHub** is a lightweight Flask microservice that performs **one-shot face verification** using a **Siamese Neural Network**. It compares a live face image against a set of enrolled verification images and determines whether the person is who they claim to be.

It is designed to act as the **biometric authentication layer** for the [Medify Hospital Management System](https://github.com/safalsingh1/Medify_backend) — enabling secure, contactless identity verification for doctors and staff.

> 📝 **Note:** The trained model (`siamesemodel.h5`) is stored separately and must be placed in the project root before running.

---

## 🧬 How It Works

```
Input Image (URL or POST body)
        │
        ▼
  Preprocess (resize to 100×100, normalize 0–1)
        │
        ▼
  Siamese Network (L1 Distance metric)
  ┌─────────────────────────────────────┐
  │  Input Image  ──►  Embedding       │
  │  Verification ──►  Embedding       │
  │       L1 Distance (custom layer)   │
  └─────────────────────────────────────┘
        │
        ▼
  Detection Score (per image pair)
        │
        ▼
  Verification Score = detected / total_images
        │
        ▼
  verified = True / False
```

The model uses **L1 absolute difference** between twin network embeddings (via a custom `L1Dist` Keras layer) to measure facial similarity. A verification decision is made by checking whether enough image pairs exceed a detection threshold.

---

## 📁 Project Structure

```
NeuroHub/
│
├── api.py               ← Main Flask app — URL-based verification API
├── app2.py              ← Alternative Flask app — local filesystem verification
├── faceid.py            ← Face capture / enrollment logic
├── get.py               ← Data fetching utilities
├── layers.py            ← Custom L1Dist Keras layer definition
├── requirement.txt      ← Python dependencies
├── siamesemodel.h5      ← (NOT in repo) Trained Siamese model — add manually
└── application_data/
    └── verification_images/  ← (Used by app2.py) Enrolled face images
```

### File Descriptions

| File | Description |
|------|-------------|
| `api.py` | Primary production API. Fetches images from URLs via a companion backend API (`/loginimage`, `/getter`), runs them through the model, and returns a verification result. |
| `app2.py` | Standalone verification endpoint `/verify`. Accepts an image URL in the POST body and compares against locally stored verification images. |
| `faceid.py` | Handles local face capture for enrollment (saving verification images). |
| `get.py` | Utility functions for data retrieval from external APIs. |
| `layers.py` | Defines the custom `L1Dist` layer required to load the Siamese model. |

---

## 📦 Requirements

- Python 3.8+
- pip

### Dependencies

```
Flask
requests
Pillow
tensorflow
numpy
```

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/safalsingh1/NeuroHub.git
cd NeuroHub
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirement.txt
```

---

## 🧠 Model Setup

The Siamese model (`siamesemodel.h5`) is **not included in this repository** (stored separately due to file size). You must obtain the trained model file and place it in the **project root**:

```
NeuroHub/
├── siamesemodel.h5   ← Place here
├── api.py
└── ...
```

The model is loaded at startup:
```python
model = tf.keras.models.load_model('siamesemodel.h5', custom_objects={'L1Dist': L1Dist})
```

> The `L1Dist` custom layer is defined in `layers.py` and is **required** for the model to load correctly.

---

## ▶️ Running the Server

### Run the URL-based API (`api.py`)

```bash
python api.py
```

The server starts on: `http://0.0.0.0:5000`

### Run the local filesystem API (`app2.py`)

```bash
python app2.py
```

> For `app2.py`, ensure the `application_data/verification_images/` directory exists and contains enrolled face images.

---

## 📡 API Reference

### `GET /get_data` — Verify via external image URLs

Fetches images from a companion backend API and verifies identity.

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `groupId` | string | ID of the login image group (the image to verify) |
| `groupLoginId` | string | ID of the enrolled images group (the known face set) |
| `url` | string | Base URL of the companion backend API |

**Example Request:**
```http
GET /get_data?groupId=abc123&groupLoginId=xyz789&url=https://hms-backend-7ub2.onrender.com
```

**Example Response:**
```json
{
  "detection": 4,
  "verification": 0.8,
  "verified": true,
  "imageurks": ["https://...", "https://..."]
}
```

---

### `POST /verify` — Verify via image URL (local images)

Accepts an image URL, fetches it, and compares it against locally stored verification images.

**Request Body:**
```json
{
  "image_url": "https://example.com/face.jpg"
}
```

**Example Response:**
```json
{
  "detection": 3,
  "verification": 0.75,
  "verified": true
}
```

---

### `GET /hello` — Health Check

```http
GET /hello
```

**Response:** `Hello` (HTTP 200)

---

## ⚖️ Thresholds Explained

The verification decision uses two configurable thresholds:

| Threshold | Default (api.py) | Default (app2.py) | Meaning |
|-----------|-----------------|-------------------|---------|
| `detection_threshold` | `0.3` | `0.4` | Minimum model score for a single image pair to count as a "detection" |
| `verification_threshold` | `0.4` | `0.6` | Minimum ratio of detected pairs to total enrolled images for final `verified=True` |

**Formula:**
```python
detection = count(results > detection_threshold)
verification = detection / total_enrolled_images
verified = verification > verification_threshold
```

You can tune these values to balance **security** (false rejection) vs. **usability** (false acceptance).

---

## 🏥 Integration with Medify

NeuroHub is designed to plug into the **Medify Hospital Management System** as a biometric authentication layer:

```
User Login Request
      │
      ▼
Medify Backend (Node.js/Express)
      │ ← Calls NeuroHub /get_data with the user's face image
      ▼
NeuroHub (Flask/TensorFlow)
      │ ← Returns { verified: true/false }
      ▼
Medify Backend grants or denies access
```

The companion Medify backend exposes two endpoints that NeuroHub consumes:
- `/loginimage?groupId=...` — Returns the user's registered login image
- `/getter?groupId=...` — Returns the set of enrolled verification images

---

## 📄 License

This project is licensed under the **MIT License**.

---

<p align="center">Made with 🧠 by <a href="https://github.com/safalsingh1">Safal Singh</a></p>
