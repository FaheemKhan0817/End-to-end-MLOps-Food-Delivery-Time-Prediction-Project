
<h1 align="center">🚀 End-to-End MLOps Food Delivery Time Prediction Project 🍔📦</h1>

<p align="center">
  <img src="static/favicon.ico" width="80" alt="Project Logo"/>
</p>

<p align="center">
  <em>Predicting food delivery times using a fully automated MLOps pipeline with real-time monitoring and cloud deployment</em>
</p>

---

## 📌 Overview

This repository contains a **production-grade MLOps pipeline** for predicting food delivery times. The project demonstrates the complete machine learning lifecycle from data preprocessing and training to deployment and monitoring, following best practices in MLOps.  
It is designed for **scalability, observability**, and **automation** using cutting-edge tools like Docker, GitHub Actions, and AWS.

---

## 🧭 Project Structure

```
End-to-end-MLOps-Food-Delivery-Time-Prediction-Project/
├── .github/workflows/            # CI/CD pipeline definitions
│   └── deploy.yml
├── artifacts/models/             # Trained model and scaler
├── config/                       # Configuration files
├── logs/                         # Application and training logs
├── notebooks/                    # EDA and experimentation
├── pipeline/                     # Training pipeline scripts
├── src/                          # Source code for API and utilities
├── app.py                        # Flask API entrypoint
├── Dockerfile                    # Docker image definition
└── requirements.txt              # Project dependencies
```

---

## 🛠️ Technologies Used

- 🐍 **Python 3.10+**
- ⚙️ **Flask** – RESTful API
- 📦 **XGBoost** – Model for regression
- 🧠 **Scikit-learn**, **Pandas**, **NumPy** – Data wrangling & model evaluation
- 🔧 **Redis** – Real-time feature store
- 🧪 **Alibi Detect** – Data drift monitoring
- 🐳 **Docker** – Containerization
- ✅ **GitHub Actions** – CI/CD automation
- ☁️ **AWS Elastic Beanstalk** – Cloud deployment

---

## ⚙️ Setup Instructions

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/End-to-end-MLOps-Food-Delivery-Time-Prediction-Project.git
cd End-to-end-MLOps-Food-Delivery-Time-Prediction-Project

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run Redis locally (optional for testing)
docker run -p 6379:6379 redis

# 5. Start the Flask app
python app.py
```

---

## 🧪 Training the Model

To run the full training pipeline:

```bash
python pipeline/training_pipeline.py
```

Artifacts (model, scaler) will be saved in the `artifacts/models/` directory.

---

## 🔍 Model Monitoring

- Alibi Detect is integrated for monitoring **data drift**.
- Redis is used as a **feature store** to track incoming requests.
- Logging captures model performance and request metadata.

---

## 🔄 CI/CD Pipeline

- **Trigger:** Code pushed to the `main` branch
- **Steps:** Lint → Test → Build Docker Image → Deploy to AWS
- Defined in `.github/workflows/deploy.yml`

---

## 🚀 Deployment

This app is deployed on **AWS Elastic Beanstalk** using Docker.  
You can deploy it locally using:

```bash
docker build -t delivery-time-predictor .
docker run -p 5000:5000 delivery-time-predictor
```

Then go to [http://localhost:5000](http://localhost:5000) in your browser.

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgements

- Inspired by real-world food delivery use-cases
- Tools by Open Source communities ❤️
