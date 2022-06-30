
# Import needed packages and functions

import argparse
import os
from io import StringIO
import json

import requests
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, confusion_matrix
import sklearn.ensemble

from azureml.core import Datastore, Environment, Webservice
from azureml.core.workspace import Workspace
from azureml.core.experiment import Experiment
from azureml.core.dataset import Dataset
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.compute_target import ComputeTargetException
from azureml.core.run import Run
from azureml.core.model import Model, InferenceConfig
from azureml.core.webservice import LocalWebservice, AciWebservice
import mlflow

# List of the column / feature names for the used Adult dataset
COLUMNS = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'martial-status', 'occupation',
    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
    'income']

# List of the names of categorical features
CATEGORICAL_COLUMNS = ['workclass', 'education', 'martial-status', 'occupation', 'relationship',
    'race', 'sex', 'native-country', 'income']

# Name of the compute cluster to be created
COMPUTE_CLUSTER_NAME = 'ndeg-prj3-clust'

# Name and path of the best model found by hyperdrive
HYPERDRIVE_MODEL_FILE = 'best_model_hyperdrive.pkl'
HYPERDRIVE_MODEL_PATH = './outputs/' + HYPERDRIVE_MODEL_FILE

# Name and path of the best model found by AutoML
AUTOML_MODEL_FILE = 'best_model_automl.pkl'
AUTOML_MODEL_PATH = './outputs/' + AUTOML_MODEL_FILE


# Function for cleaning the data
def clean_data(df):
    df = df.replace(to_replace='\\?', value=np.nan, regex=True) # Replace ? by np.nan for dropping missing values
    df = df.replace(to_replace='\s+', value='', regex=True)     # Remove white space in categorical feature values
    df = df.replace(to_replace='\\.', value='', regex=True)     # Remove dots from categorical feature values
    for column in CATEGORICAL_COLUMNS:
        df[column] = df[column].astype(str)  # Turn the values of all categorical columns into strings
    return df


# Function for replacing all categorical values by integers
def label_encode_data(train_df, test_df):
    for column in CATEGORICAL_COLUMNS:
        label_encoder = LabelEncoder()
        label_encoder.fit(train_df[column]) # Fit the encoder on the training data
        train_df[column] = label_encoder.transform(train_df[column]) # Apply the fitted encoder to the training data
        test_df[column] = label_encoder.transform(test_df[column])   # Apply the fitted encoder to the test data
    return train_df, test_df


# Function for preprocessing the data
def preprocess_data(train_csv_file, test_csv_file, label_encode=False):
    # Turn train and test data from CSV into pandas dataframes 
    train_df = pd.read_csv(train_csv_file, delimiter=',', header=None, names=COLUMNS)
    test_df = pd.read_csv(test_csv_file, delimiter=',', header=None, names=COLUMNS)
    # Clean train and test data
    train_df = clean_data(train_df)
    test_df = clean_data(test_df)
    if label_encode:
        # Encode categorical values as integers if requested
        train_df, test_df = label_encode_data(train_df, test_df)
    return train_df, test_df


# Helper function for getting the current Azure workspace
def get_workspace():
    ws = Workspace.from_config()
    return ws


# Function for getting and installing the Adult dataset from UCI
def get_data(ws=None, suffix='automl'):
    
    train_ds = None
    test_ds = None

    if ws is None:
        ws = get_workspace()

    label_encode=True
    if suffix == 'automl':
        label_encode=False # label encoding not needed for AutoML
        
    # Construct names of the training and test set based on the provided suffix ("hyperd" or "automl")
    train_name = f'adult_train_{suffix}'
    test_name = f'adult_test_{suffix}'
        
    if train_name in ws.datasets.keys() and test_name in ws.datasets.keys():
        # If the data is already registered in Azure, load it from the workspace
        print('Loading datasets from workspace ...')
        train_ds = ws.datasets[train_name]
        test_ds = ws.datasets[test_name]
 
    else:
        # Otherwise load it from the web and register it in the workspace
        print('Loading datasets from web and registering them in workspace ...')
        train_file = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
        train_file_resp = requests.get(train_file) # download training data from URL
        test_file = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
        test_file_resp = requests.get(test_file) # download test data from URL
        test_text = '\n'.join(test_file_resp.text.split('\n')[1:]) # remove first line from test data
        # Preprocess the data
        train_df, test_df = preprocess_data(StringIO(train_file_resp.text), StringIO(test_text), label_encode)
        # Get the blob store
        datastore = Datastore.get(ws, 'workspaceblobstore')
        if train_name not in ws.datasets.keys():
            # register the training set if not already done so
            train_ds = Dataset.Tabular.register_pandas_dataframe(train_df,
                datastore, train_name, show_progress=True)
        if test_name not in ws.datasets.keys():
            # register the test set if not already done so
            test_ds = Dataset.Tabular.register_pandas_dataframe(test_df,
                datastore, test_name, show_progress=True)
            
    return train_ds, test_ds


# Function for getting access to an existing compute cluster or creating one
def get_compute_cluster():

    ws = get_workspace()

    try:
        # Try to find an existing cluster and return it if it exists
        compute_cluster = ComputeTarget(workspace=ws, name=COMPUTE_CLUSTER_NAME)
        print('Found existing cluster, use it.')
    except ComputeTargetException:
        # Otherwise create a new compute cluster with four nodes
        compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D2_V2',# for GPU, use "STANDARD_NC6"
                                                               #vm_priority = 'lowpriority', # optional
                                                               max_nodes=4)
        compute_cluster = ComputeTarget.create(ws, COMPUTE_CLUSTER_NAME, compute_config)
        compute_cluster.wait_for_completion(show_output=True)

    return compute_cluster


# Get an environment for training and running scikit-learn models on remote clusters 
def get_hyperd_environment():
    hyperd_env = Environment.from_conda_specification(name='hyperd-env', file_path='conda_dependencies_hyperd.yml')
    return hyperd_env


# Get an environment for training and running AutoML models on remote clusters
def get_automl_environment():
    automl_env = Environment.from_conda_specification(name='automl-env', file_path='conda_dependencies_automl.yml')
    return automl_env


# Run a hyperparameter optimization using Hyperdrive
def run_hyperd(hd_config):
    
    from azureml.widgets import RunDetails
    
    ws = get_workspace()

    exp = Experiment(workspace=ws, name='adult-hyperd')
    run = exp.start_logging()

    if "training" not in os.listdir():
        os.mkdir("./training")

    # Submit hyperdrive run and show its details
    hyperdrive_run = exp.submit(config=hd_config)
    RunDetails(hyperdrive_run).show()
    hyperdrive_run.wait_for_completion(show_output=True)

    # When finished, get the best model from the best run
    best_run = hyperdrive_run.get_best_run_by_primary_metric()
    best_run_metrics = best_run.get_metrics()
    parameter_values = best_run.get_details()['runDefinition']['arguments']

    print('Best run ID:', best_run.id)
    print('Accuracy:', best_run_metrics['accuracy'])
    print('Parameters:', parameter_values)

    # Download the best model
    best_run.download_file('outputs/model.pkl', HYPERDRIVE_MODEL_PATH, _validate_checksum=True)
    

# Print information about the best model and test it on the test data
def show_and_test_local_hyperd_model(test_ds):
    model = joblib.load(HYPERDRIVE_MODEL_PATH)
    print(model) # show information about the best model
    # Prepare test set for prediction
    test_df = test_ds.to_pandas_dataframe()
    X_test = test_df.drop(['income'], axis=1).to_numpy()
    y_test = test_df['income'].to_numpy()
    y_pred = model.predict(X_test)
    # Build a confusion matrix based on the true labels and predictions
    cm = confusion_matrix(y_test, y_pred)
    # Calculate model's accuracy on the test data
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)
    return pd.DataFrame(cm).style.background_gradient(cmap='Blues', low=0, high=0.9)


# Function for registering and deploying the model found by hyperdrive
def deploy_hyperd_model(model):
    
    ws = get_workspace()
    
    # Get a scikit-learn environment for running the model
    hyperd_env = get_hyperd_environment()

    # Specify entry script under ./source_dir in the inference config
    inference_config = InferenceConfig(
        environment=hyperd_env,
        source_directory='./source_dir',
        entry_script='./predict_hyperd.py',
        )

    # Configure a compute instance for model deployment
    deployment_config = AciWebservice.deploy_configuration(
        cpu_cores=0.5, memory_gb=1, auth_enabled=True
        )

    # Deploy model to a compute instance
    service = Model.deploy(ws, 'adult-hyperd-service', [model],
        inference_config, deployment_config, overwrite=True)
    service.wait_for_deployment(show_output=True)

    # Print logs to see if deployment was successful
    print(service.get_logs())


# Cast values in test input to basic types for JSON encoding
def cast_hyperd_test_input(test_input):
    cast_test_input = {}
    cast_test_input['age'] = int(test_input['age'])
    cast_test_input['workclass'] = int(test_input['workclass'])
    cast_test_input['fnlwgt'] = int(test_input['fnlwgt'])    
    cast_test_input['education'] = int(test_input['education'])
    cast_test_input['education-num'] = int(test_input['education-num'])
    cast_test_input['martial-status'] = int(test_input['martial-status'])
    cast_test_input['occupation'] = int(test_input['occupation'])
    cast_test_input['relationship'] = int(test_input['relationship'])
    cast_test_input['race'] = int(test_input['race'])
    cast_test_input['sex'] = int(test_input['sex'])
    cast_test_input['capital-gain'] = int(test_input['capital-gain'])
    cast_test_input['capital-loss'] = int(test_input['capital-loss'])
    cast_test_input['hours-per-week'] = int(test_input['hours-per-week'])
    cast_test_input['native-country'] = int(test_input['native-country'])
    return cast_test_input


# Create a test input from a row in the Adult test dataset
def create_hyperd_test_input(test_ds, row):
    test_df = test_ds.to_pandas_dataframe().iloc[row]
    test_label = int(test_df['income']) # remember label
    del test_df['income'] # delete label column
    test_input = cast_hyperd_test_input(test_df)
    return test_input, test_label # return test input and label
    

# Test the deployed random forst model with a row from the Adult test set
def test_deployed_hyperd_model(test_ds, row):
    
    ws = get_workspace()
    
    # Get web service
    service = Webservice(workspace=ws, name='adult-hyperd-service')
    # Get URI and authorization keys
    scoring_uri = service.scoring_uri
    key, _ = service.get_keys()

    # Construct HTTP request headers
    headers = {"Content-Type": "application/json"}
    headers['Authorization'] = f'Bearer {key}'
    # Create input data from the specified row of the Adult test set
    test_input, test_label = create_hyperd_test_input(test_ds, row)
    # JSON encode the test input
    data = json.dumps(test_input)
    # Send HTTP post request with data and headers
    response = requests.post(scoring_uri, data=data, headers=headers)
    
    # Print label and prediction (response) for comparison
    print('label:', test_label, ', prediction:', int(response.json()))
 
    
def run_automl():
    
    from azureml.train.automl import AutoMLConfig
    from azureml.widgets import RunDetails

    ws = get_workspace()
    
    # Create an AutoML experiment
    exp = Experiment(workspace=ws, name='adult-automl')
    run = exp.start_logging()
    
    # Get a compute cluster (or create one)
    compute_cluster = get_compute_cluster()
    
    # Get the training data
    train_ds, _ = get_data()
    
    # Configure AutoML to run for a maximum of 30 minutes
    # Use 'accuracy' as the primary metric, predict the income,
    # use 5-fold cross validation for performance evaluation
    automl_config = AutoMLConfig(
        experiment_timeout_minutes=30,
        task='classification',
        primary_metric='accuracy',
        training_data=train_ds,
        label_column_name='income',
        n_cross_validations=5,
        compute_target=compute_cluster)
    
    # Submit AutoML run
    automl_run = exp.submit(automl_config)
    RunDetails(automl_run).show()
    automl_run.wait_for_completion(show_output=True)
    
    # Get the best run and model
    best_run, best_model = automl_run.get_output()
    best_run_metrics = best_run.get_metrics()

    print('Best run ID:', best_run.id)
    print('Accuracy:', best_run_metrics['accuracy'])

    # Save the best AutoML model
    joblib.dump(best_model, AUTOML_MODEL_PATH)
    

def show_and_test_local_automl_model(test_ds):
    model = joblib.load(AUTOML_MODEL_PATH)
    # Show steps in the AutoML model
    for step in model.steps:
        print(step)
    # Construct test data
    test_df = test_ds.to_pandas_dataframe()
    X_test = test_df.drop(['income'], axis=1)
    y_test = test_df['income']
    y_pred = model.predict(X_test)
    # Calculate confusion matrix based on the true labels and predictions
    cm = confusion_matrix(y_test, y_pred)
    # Calculate the model's accuracy on the Adult test set
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)
    return pd.DataFrame(cm).style.background_gradient(cmap='Blues', low=0, high=0.9)


# Function for registering and deploying the best AutoML model
def register_and_deploy_automl_model():
    
    ws = get_workspace()
    
    # Register best model in workspace
    model = Model.register(ws, model_name='adult-automl-model',
        model_path=AUTOML_MODEL_PATH)

    # Get an environment for AutoML
    automl_env = get_automl_environment()

    # Provide an entry script under ./source_dir in the inference config
    inference_config = InferenceConfig(
        environment=automl_env,
        source_directory='./source_dir',
        entry_script='./predict_automl.py',
        )

    # Configure deployment on a compute instance
    deployment_config = AciWebservice.deploy_configuration(
        cpu_cores=0.5, memory_gb=1, auth_enabled=True
        )

    # Deploy model on the compute instance
    service = Model.deploy(ws, 'adult-automl-service', [model],
        inference_config, deployment_config, overwrite=True)
    service.wait_for_deployment(show_output=True)

    # Get logs from the compute instance to see if it is running
    print(service.get_logs())
    
    return service

    
# Cast all field values from test input to basic types for JSON encoding
def cast_automl_test_input(test_input):
    cast_test_input = {}
    cast_test_input['age'] = int(test_input['age'])
    cast_test_input['workclass'] = str(test_input['workclass'])
    cast_test_input['fnlwgt'] = int(test_input['fnlwgt'])    
    cast_test_input['education'] = str(test_input['education'])
    cast_test_input['education-num'] = int(test_input['education-num'])
    cast_test_input['martial-status'] = str(test_input['martial-status'])
    cast_test_input['occupation'] = str(test_input['occupation'])
    cast_test_input['relationship'] = str(test_input['relationship'])
    cast_test_input['race'] = str(test_input['race'])
    cast_test_input['sex'] = str(test_input['sex'])
    cast_test_input['capital-gain'] = int(test_input['capital-gain'])
    cast_test_input['capital-loss'] = int(test_input['capital-loss'])
    cast_test_input['hours-per-week'] = int(test_input['hours-per-week'])
    cast_test_input['native-country'] = str(test_input['native-country'])
    return cast_test_input


# Takes a row from the Adult test set and creates an input for the deployed model from it
def create_automl_test_input(test_ds, row):
    test_df = test_ds.to_pandas_dataframe().iloc[row]
    test_label = test_df['income'] # remember label
    del test_df['income'] # drop label column
    test_input = cast_automl_test_input(test_df) # construct input data from dataframe
    return test_input, test_label # return input data and label


# Test the deployed AutoML  model with a row from the Adult test set
def test_deployed_automl_model(test_ds, row, service=None):
    
    ws = get_workspace()
    
    if service is None:
        # Get web service (endpoint) by name
        service = Webservice(workspace=ws, name='adult-automl-service')
    # Get scoring URI and authorization keys
    scoring_uri = service.scoring_uri
    key, _ = service.get_keys()

    # Construct request headers
    headers = {"Content-Type": "application/json"}
    headers['Authorization'] = f'Bearer {key}'
    # Create test input for specified row in Adult test set
    test_input, test_label = create_automl_test_input(test_ds, row)
    # JSON encode input data
    data = json.dumps(test_input)
    # Send headers and data via HTTP post request
    response = requests.post(scoring_uri, data=data, headers=headers)
    
    # Print label and received prediction for comparison
    print('label:', test_label, ', prediction:', str(response.json()))
    

# Delete cluster, web service and registered model
def clean_up(automl=False):
    
    model_type = 'hyperd'
    if automl:
        model_type = 'automl'
    
    print('Delete compute cluster ...')
    compute_cluster = get_compute_cluster()
    compute_cluster.delete()
    
    ws = get_workspace()
    print('Delete web service ...')
    service = Webservice(workspace=ws, name=f'adult-{model_type}-service')
    print(service.get_logs()) # print logs before deleting the service
    service.delete()
    
    print('Delete registered models ...')
    for model in Model.list(ws):
        if model_type in model.name:
            Model(ws, name=model.name, version=model.version).delete()
    

# Main part for training a random forst model with hyperdrive    

def main():

    # Parse arguments given to the script
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_estimators', type=int, default=10, help='Number of estimators in the ensemble')
    parser.add_argument('--max_depth', type=int, default=5, help='Maximum depth of the trees involved')
    parser.add_argument('--max_features', type=int, default=14, help='Maximum number of features per tree')
    parser.add_argument('--min_samples_leaf', type=float, default=0.05, help='Minimum number of samples per leaf')

    args = parser.parse_args()
    
    # Get the context in which this script is run
    
    run = Run.get_context()

    # Read parameters provided by HyperDrive run
    run.log('number of estimators', int(args.n_estimators))
    run.log('max depth', int(args.max_depth))
    run.log('max features', int(args.max_features))
    run.log('min samples leaf', float(args.min_samples_leaf))

    # Get training data from workspace and prepare it for training
    ws = run.experiment.workspace
    train_ds, _ = get_data(ws, suffix='hyperd')
    train_df = train_ds.to_pandas_dataframe()
    X_train = train_df.drop(['income'], axis=1).to_numpy()
    y_train = train_df['income'].to_numpy()
    
    # Create a random forest classifier with the provided hyperparameters
    clf = sklearn.ensemble.RandomForestClassifier(n_estimators=args.n_estimators,
        max_depth=args.max_depth, max_features=args.max_features,
        min_samples_leaf=args.min_samples_leaf, random_state=42)
    
    # Fit the classifier and save the resulting model for potential later use
    model_for_all_data = clf.fit(X_train, y_train)
    joblib.dump(model_for_all_data, './outputs/model.pkl')
    
    # Validate the model using 5-fold cross validation
    cv_results = cross_validate(clf, X_train, y_train, cv=5, scoring='accuracy', verbose=2)
    
    accuracy = float(cv_results['test_score'].mean())
    
    # Log the accuracy from the cross validation
    run.log("accuracy", accuracy)
    
    # Log metric with mlflow
    #mlflow.log_metric('accuracy', accuracy)
    # This does not work - see https://github.com/Azure/azure-sdk-for-python/issues/23563 - one commenter says that either the
    # SDK or mlflow should be used, not both - I am already using the SDK!
    # Note also that I am using SDK version 1.42.0 - not v2!


if __name__ == '__main__':
    main()

 