# train_pipeline.py
import pandas as pd
import numpy as np
import os
import sys

from src.exception import CustomException
from src.utils import save_object, evaluate_models
from src.components.model_trainer import ModelTrainerConfig

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

def train():
    try:
        # Load preprocessed data
        train_path = os.path.join("artifacts", "train.csv")
        test_path = os.path.join("artifacts", "test.csv")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        X_train = train_df.iloc[:, :-1]
        y_train = train_df.iloc[:, -1]
        X_test = test_df.iloc[:, :-1]
        y_test = test_df.iloc[:, -1]

        models = {
            "Random Forest": RandomForestClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "Logistic Regression": LogisticRegression(),
            "K-Neighbors Classifier": KNeighborsClassifier(),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        }

        params = {
            "Decision Tree": {
                'criterion': ['gini', 'entropy'],
                'max_depth': [5, 10, 20]
            },
            "Random Forest": {
                'n_estimators': [50, 100, 200]
            },
            "Gradient Boosting": {
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [50, 100, 200]
            },
            "Logistic Regression": {
                'C': [0.1, 1.0, 10.0],
                'solver': ['liblinear']
            },
            "K-Neighbors Classifier": {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance']
            },
            "XGBoost": {
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [50, 100, 200]
            },
        }

        model_report, trained_models = evaluate_models(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            models=models, param=params
        )

        best_model_score = max(model_report.values())
        best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
        best_model = trained_models[best_model_name]

        if best_model_score < 0.6:
            raise CustomException("No good model found with F1 score above 60%")

        print(f"Best model: {best_model_name} | Score: {best_model_score}")

        save_object(
            file_path=ModelTrainerConfig.trained_model_file_path,
            obj=best_model
        )

    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    train()
