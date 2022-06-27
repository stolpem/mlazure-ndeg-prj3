import json
import os.path

import numpy as np
import joblib


def init():
    print('This is init')
    global model
    model = joblib.load(os.path.join(os.getenv("AZUREML_MODEL_DIR"), 'best_hyperdrive_model.pkl')
    print(model)


def run(data):
    input_data = json.loads(data)
    print(f"received data {input_data}")
    input_data_array = []
    for key in input_data.keys():
        input_data_array.append(input_data[key])
    prediction = model.predict(np.array(input_data_array, nddim=2))
    return int(prediction[0])
