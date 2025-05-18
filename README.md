# 💳 Credit Card Fraud Detection Web App

A machine learning-powered Flask web application that detects fraudulent credit card transactions. This app takes user input for transaction features, processes the input through a pre-trained ML model, and returns a prediction on whether the transaction is **fraudulent** or **legitimate**.

---

## 🚀 Features

- Clean user-friendly UI (HTML/CSS + Bootstrap)
- Real-time fraud detection based on user inputs
- Trained model using SMOTE + GridSearchCV + XGBoost
- Modular and scalable ML pipeline
- Docker-friendly project structure (optional)

---

## 📊 Dataset

The app is trained on the [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle.  
It contains anonymized PCA-transformed features (`V1` to `V28`), `Time`, and `Amount`.

---

## 🧠 Machine Learning Pipeline

- **Data Preprocessing**: Handling imbalanced data using SMOTE, scaling, and feature selection
- **Model Training**: Multiple classifiers tested, best model selected via GridSearchCV
- **Model Used**: XGBoostClassifier with optimal parameters
- **Evaluation Metric**: F1-Score

---

## 🖥️ Web Interface

- Built with **Flask** backend
- Accepts form input for all 30 features
- Renders prediction with styled output
