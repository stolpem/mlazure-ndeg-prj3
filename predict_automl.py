import json
import os.path

import pandas as pd
import numpy as np
import joblib


def init():
    print('This is init')
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'best_model_automl.pkl')
    model = joblib.load(model_path)
    print(model)


def run(data):
    input_data = json.loads(data)
    print(f'received data {input_data}')
    input_data_pd = {}
    for key in input_data:
        input_data_pd[key] = [input_data[key]]
    input_data_df = pd.DataFrame(input_data_pd)
    prediction = model.predict(input_data_df)
    return str(prediction[0])