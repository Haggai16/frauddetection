import os
import sys
import dill

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param: dict):
    try:
        report = {}
        trained_models = {}

        for model_name, model in models.items():
            para = param.get(model_name, {})

            gs = GridSearchCV(model, para, cv=5, verbose=2, scoring='f1', n_jobs=-1)
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_
            trained_models[model_name] = best_model

            y_test_pred = best_model.predict(X_test)
            test_f1 = f1_score(y_test, y_test_pred)

            report[model_name] = test_f1

        return report, trained_models

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
