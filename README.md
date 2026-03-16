# Fraud Detection ML API

A Machine Learning based **Credit Card Fraud Detection System** built with **PyTorch** and deployed using **FastAPI**.

This project trains a neural network on a real-world credit card transaction dataset and exposes a REST API to predict fraud probability.

---

# 🚀 Features

* Neural Network Fraud Detection Model (PyTorch)
* Handles highly **imbalanced fraud datasets**
* Feature scaling using **StandardScaler**
* Fraud probability scoring
* **FastAPI REST API** for inference
* Swagger UI for testing
* Model + scaler persistence

---

# 🧠 How It Works

Pipeline:

```
Dataset
   ↓
Data Preprocessing
   ↓
Neural Network Training
   ↓
Model Evaluation
   ↓
Save Model (.pth)
   ↓
FastAPI Inference API
   ↓
Fraud Probability Prediction
```

---

# 📊 Dataset

Dataset used:

**Credit Card Fraud Detection Dataset**

Features:

* 30 transaction features
* Highly imbalanced dataset (~0.17% fraud)

Target:

```
Class = 0 → Normal transaction
Class = 1 → Fraud transaction
```

---

# 🏗 Project Structure

```
fraud-detection-ml-api/

│
├── creditcard.csv        # Dataset
├── main.py               # Model training pipeline
├── predict.py            # Local prediction script
├── app.py                # FastAPI ML service
├── fraud_model.pth       # Trained model
├── scaler.pkl            # Feature scaler
├── requirements.txt      # Dependencies
└── README.md
```

---

# ⚙️ Installation

Clone the repository:

```
git clone https://github.com/yourusername/fraud-detection-ml-api.git
cd fraud-detection-ml-api
```

Create virtual environment:

```
python -m venv venv
```

Activate environment:

Windows:

```
venv\Scripts\activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# 🧠 Train Model

Run:

```
python main.py
```

This will:

* Train the neural network
* Evaluate model performance
* Save:

```
fraud_model.pth
scaler.pkl
```

---

# 🔮 Run Prediction Script

```
python predict.py
```

This loads the trained model and predicts fraud probability.

---

# 🚀 Run FastAPI Service

Start API server:

```
uvicorn app:app --reload
```

Open API docs:

```
http://127.0.0.1:8000/docs
```

Example request:

```
POST /predict
```

Input:

```json
{
 "features": [30 transaction values]
}
```

Response:

```json
{
 "fraud_probability": 0.032,
 "fraud": false
}
```

---

# 🧠 Technologies Used

* PyTorch
* Scikit-learn
* FastAPI
* Uvicorn
* NumPy
* Pandas

---

# 📈 Future Improvements

* Feature engineering
* Transformer based fraud detection
* Batch prediction endpoint
* Docker containerization
* Real-time fraud detection pipeline
* Model monitoring

---

# 👨‍💻 Author

Abhishek
Backend Developer | ServiceNow Developer | AI/ML Enthusiast
