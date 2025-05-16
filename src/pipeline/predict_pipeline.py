# predict_pipeline.py
import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class CustomData:
    def __init__(self, **kwargs):
        self.data = kwargs

    def get_data_as_dataframe(self):
        try:
            df = pd.DataFrame([self.data])
            return df
        except Exception as e:
            raise Exception(f"Error in CustomData.get_data_as_dataframe: {e}")


class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    def predict(self, input_df: pd.DataFrame):
        try:
            model = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)

            data_scaled = preprocessor.transform(input_df)
            prediction = model.predict(data_scaled)
            return prediction

        except Exception as e:
            raise CustomException(e, sys)
