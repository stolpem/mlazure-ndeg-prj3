
from io import StringIO

import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from azureml.core import Datastore
from azureml.core.dataset import Dataset


COLUMNS = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'martial-status', 'occupation',
    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
    'income']
            
CATEGORICAL_COLUMNS = ['workclass', 'education', 'martial-status', 'occupation', 'relationship',
    'race', 'sex', 'native-country', 'income'] 


def clean_data(df):
    df = df.replace(to_replace='\\?', value=np.nan, regex=True)
    df = df.replace(to_replace='\s+', value='', regex=True)
    df = df.replace(to_replace='\\.', value='', regex=True)
    for column in CATEGORICAL_COLUMNS:
        df[column] = df[column].astype(str)
    return df


def label_encode_data(train_df, test_df):
    for column in CATEGORICAL_COLUMNS:
        label_encoder = LabelEncoder()
        label_encoder.fit(train_df[column])
        train_df[column] = label_encoder.transform(train_df[column])
        test_df[column] = label_encoder.transform(test_df[column])
    return train_df, test_df


def preprocess_data(train_csv_file, test_csv_file, label_encode=False):
    train_df = pd.read_csv(train_csv_file, delimiter=',', header=None, names=COLUMNS)
    test_df = pd.read_csv(test_csv_file, delimiter=',', header=None, names=COLUMNS)
    train_df = clean_data(train_df)
    test_df = clean_data(test_df)
    if label_encode:
        train_df, test_df = label_encode_data(train_df, test_df)
    return train_df, test_df


def get_hyperd_data(ws):

    if 'adult_train_hyperd' in ws.datasets.keys() and 'adult_test_hyperd' in ws.datasets.keys():
        print('Loading datasets from workspace ...')
        adult_train = ws.datasets['adult_train_hyperd']
        adult_test = ws.datasets['adult_test_hyperd']
 
    else:
        print('Loading datasets from web and registering them in workspace ...')
        train_file = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
        train_file_resp = requests.get(train_file)
        test_file = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
        test_file_resp = requests.get(test_file)
        test_text = '\n'.join(test_file_resp.text.split('\n')[1:]) # remove first line
        train_df, test_df = preprocess_data(StringIO(train_file_resp.text), StringIO(test_text), label_encode=True)
        datastore = Datastore.get(ws, 'workspaceblobstore')
        if 'adult_train_hyperd' not in ws.datasets.keys():
            train_ds = Dataset.Tabular.register_pandas_dataframe(train_df, datastore, 'adult_train_hyperd', show_progress=True)
        if 'adult_test_hyperd' not in ws.datasets.keys():
            test_ds = Dataset.Tabular.register_pandas_dataframe(test_df, datastore, 'adult_test_hyperd', show_progress=True)
            
    return train_ds, test_ds

