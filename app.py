from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

    if request.method == 'POST':
        try:
            data = {col: float(request.form[col]) for col in columns}
            custom_data = CustomData(**data)
            input_df = custom_data.get_data_as_dataframe()
            pipeline = PredictPipeline()
            prediction = pipeline.predict(input_df)[0]
            prediction = 'Fraudulent' if prediction == 1 else 'Legitimate'
        except Exception as e:
            prediction = f"Error during prediction: {e}"  # Display error on the page

    return render_template('index.html', prediction=prediction, columns=columns)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')