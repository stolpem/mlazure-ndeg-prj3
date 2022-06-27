import json
import os.path

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
    input_data_array = []
    for key in input_data.keys():
        input_data_array.append(input_data[key])
    prediction = model.predict(np.array(input_data_array, ndmin=2))
    return int(prediction[0])
