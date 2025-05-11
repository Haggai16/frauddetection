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


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for i in range(len(models)):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            para = param.get(model_name, {})

            gs = GridSearchCV(model, para, cv=3, scoring='f1', n_jobs=-1)
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_

            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            # You can change the metric depending on what's most important
            test_f1 = f1_score(y_test, y_test_pred)
            # Alternatively:
            # test_accuracy = accuracy_score(y_test, y_test_pred)
            # test_roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])

            report[model_name] = test_f1

        return report

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
