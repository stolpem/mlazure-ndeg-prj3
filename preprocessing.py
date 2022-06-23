import argparse
import os
from io import StringIO
import joblib

import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate
import sklearn.ensemble

from azureml.core.workspace import Workspace
from azureml.core import Datastore
from azureml.core.dataset import Dataset


COLUMNS = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'martial-status', 'occupation',
    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
    'income']
            
CATEGORICAL_COLUMNS = ['workclass', 'education', 'martial-status', 'occupation', 'relationship',
    'race', 'sex', 'native-country', 'income'] 



class Parameters:
    def __init__(self):
        pass


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


def get_hyperd_data(ws=None):
    
    train_ds = None
    test_ds = None
    
    if ws is None:
        ws = Workspace.from_config()

    if 'adult_train_hyperd' in ws.datasets.keys() and 'adult_test_hyperd' in ws.datasets.keys():
        print('Loading datasets from workspace ...')
        train_ds = ws.datasets['adult_train_hyperd']
        test_ds = ws.datasets['adult_test_hyperd']
 
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


def main():

    #parser = argparse.ArgumentParser()

    #parser.add_argument('--n_estimators', type=int, default=10, help="Number of estimators in the ensemble")
    #parser.add_argument('--max_depth', type=int, default=5, help="Maximum depth of the trees involved")
    #parser.add_argument('--max_features', type=int, default=14, help="Maximum number of features per tree")
    #parser.add_argument('--min_samples_leaf', type=float, default=0.1, help="Minimum number of samples per leaf")

    #args = parser.parse_args()
    args = Parameters()
    args.n_estimators = 30
    args.max_depth = 5
    args.max_features = 14
    args.min_samples_leaf = 0.1

    #run = Run.get_context()

    # Read parameters provided by HyperDrive run
    #run.log("Number of estimators:", np.int(args.n_estimators))
    #run.log("Max depth:", np.int(args.max_depth))
    #run.log("Max features:", np.int(args.max_features))
    #run.log("Min samples leaf:", np.float(args.min_samples_leaf))
    
    # Download data and clean it
    #ds = TabularDatasetFactory.from_delimited_files('https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv')
    #x, y = clean_data(ds)
    
    train_ds, _ = get_hyperd_data()
    train_df = train_ds.to_pandas_dataframe()
    X_train = train_df.drop(['income'], axis=1).to_numpy()
    y_train = train_df['income'].to_numpy()
    
    # Fit logistic regression model to data in training set
    clf = sklearn.ensemble.RandomForestClassifier(n_estimators=args.n_estimators,
        max_depth=args.max_depth, max_features=args.max_features,
        min_samples_leaf=args.min_samples_leaf, random_state=42)
    
    cv_results = cross_validate(clf, X_train, y_train, cv=5, scoring='accuracy')
    print(cv_results['test_score'].mean())
    
    #run.log("accuracy", np.float(cv_results['test_score'].mean()))

    # Save model for potential later use
    #joblib.dump(model_all_data, './outputs/model.pkl')

          
if __name__ == '__main__':
    main()

 