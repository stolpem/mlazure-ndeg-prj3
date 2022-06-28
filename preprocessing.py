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
from sklearn.metrics import accuracy_score
import sklearn.ensemble

from azureml.core import Datastore, Environment, ScriptRunConfig, Webservice
from azureml.core.workspace import Workspace
from azureml.core.experiment import Experiment
from azureml.core.dataset import Dataset
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.compute_target import ComputeTargetException
from azureml.core.run import Run
from azureml.core.model import Model, InferenceConfig
from azureml.core.webservice import LocalWebservice, AciWebservice


COLUMNS = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'martial-status', 'occupation',
    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
    'income']
            
CATEGORICAL_COLUMNS = ['workclass', 'education', 'martial-status', 'occupation', 'relationship',
    'race', 'sex', 'native-country', 'income']

COMPUTE_CLUSTER_NAME = 'ndeg-prj3-clust'

HYPERDRIVE_MODEL_FILE = 'best_model_hyperdrive.pkl'
HYPERDRIVE_MODEL_PATH = './outputs/' + HYPERDRIVE_MODEL_FILE

AUTOML_MODEL_FILE = 'best_model_automl.pkl'
AUTOML_MODEL_PATH = './outputs/' + AUTOML_MODEL_FILE


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


def get_workspace():
    ws = Workspace.from_config()
    return ws


def get_data(ws=None, suffix='automl'):
    
    train_ds = None
    test_ds = None

    if ws is None:
        ws = get_workspace()

    label_encode=True
    if suffix == 'automl':
        label_encode=False
        
    train_name = f'adult_train_{suffix}'
    test_name = f'adult_test_{suffix}'
        
    if train_name in ws.datasets.keys() and test_name in ws.datasets.keys():
        print('Loading datasets from workspace ...')
        train_ds = ws.datasets[train_name]
        test_ds = ws.datasets[test_name]
 
    else:
        print('Loading datasets from web and registering them in workspace ...')
        train_file = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
        train_file_resp = requests.get(train_file)
        test_file = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
        test_file_resp = requests.get(test_file)
        test_text = '\n'.join(test_file_resp.text.split('\n')[1:]) # remove first line
        train_df, test_df = preprocess_data(StringIO(train_file_resp.text), StringIO(test_text), label_encode)
        datastore = Datastore.get(ws, 'workspaceblobstore')
        if train_name not in ws.datasets.keys():
            train_ds = Dataset.Tabular.register_pandas_dataframe(train_df,
                datastore, train_name, show_progress=True)
        if test_name not in ws.datasets.keys():
            test_ds = Dataset.Tabular.register_pandas_dataframe(test_df,
                datastore, test_name, show_progress=True)
            
    return train_ds, test_ds


def get_compute_cluster():

    ws = get_workspace()

    try:
        compute_cluster = ComputeTarget(workspace=ws, name=COMPUTE_CLUSTER_NAME)
        print('Found existing cluster, use it.')
    except ComputeTargetException:
        compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D2_V2',# for GPU, use "STANDARD_NC6"
                                                               #vm_priority = 'lowpriority', # optional
                                                               max_nodes=4)
        compute_cluster = ComputeTarget.create(ws, COMPUTE_CLUSTER_NAME, compute_config)
    
    compute_cluster.wait_for_completion(show_output=True)

    return compute_cluster


def get_hyperd_environment():
    hyperd_env = Environment.from_conda_specification(name='hyperd-env', file_path='conda_dependencies_hyperd.yml')
    return hyperd_env


def get_automl_environment():
    automl_env = Environment.from_conda_specification(name='automl-env', file_path='conda_dependencies_automl.yml')
    return automl_env


def run_hyperd():

    from azureml.train.hyperdrive.run import PrimaryMetricGoal
    from azureml.train.hyperdrive.policy import BanditPolicy
    from azureml.train.hyperdrive.sampling import RandomParameterSampling
    from azureml.train.hyperdrive.runconfig import HyperDriveConfig
    from azureml.train.hyperdrive.parameter_expressions import choice, uniform
    from azureml.widgets import RunDetails
    
    ws = get_workspace()

    exp = Experiment(workspace=ws, name='adult-hyperd')
    run = exp.start_logging()

    ps = RandomParameterSampling({
        '--n_estimators': choice(range(2, 100)),
        '--max_depth': choice(range(2, 10)),
        '--max_features': choice(range(1, 14)),
        '--min_samples_leaf': uniform(0.01, 0.5)
    })

    policy = BanditPolicy(evaluation_interval=2, slack_factor=0.1) # evaluate performance every two runs,
                                                                # stop if lower than 1% point difference to
                                                                # best result in previous two runs
    if "training" not in os.listdir():
        os.mkdir("./training")

    compute_cluster = get_compute_cluster()
    hyperd_env = get_hyperd_environment()

    src = ScriptRunConfig(
        source_directory=".",
        script="preprocessing.py",
        compute_target=compute_cluster,
        environment=hyperd_env
    )

    hyperdrive_config = HyperDriveConfig(run_config=src,
        hyperparameter_sampling=ps,
        policy=policy,
        primary_metric_name='accuracy',
        primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
        max_total_runs=20,
        max_concurrent_runs=3)

    hyperdrive_run = exp.submit(config=hyperdrive_config)
    RunDetails(hyperdrive_run).show()
    hyperdrive_run.wait_for_completion(show_output=True)

    best_run = hyperdrive_run.get_best_run_by_primary_metric()
    best_run_metrics = best_run.get_metrics()
    parameter_values = best_run.get_details()['runDefinition']['arguments']

    print('Best run ID:', best_run.id)
    print('Accuracy:', best_run_metrics['accuracy'])
    print('Parameters:', parameter_values)

    best_run.download_file('outputs/model.pkl', HYPERDRIVE_MODEL_PATH, _validate_checksum=True)
    

def test_local_hyperd_model(test_ds):
    model = joblib.load(HYPERDRIVE_MODEL_PATH)
    test_df = test_ds.to_pandas_dataframe()
    X_test = test_df.drop(['income'], axis=1).to_numpy()
    y_test = test_df['income'].to_numpy()
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)
    return accuracy


def register_and_deploy_hyperd_model():
    
    ws = get_workspace()
    
    model = Model.register(ws, model_name='adult-hyperd-model',
        model_path=HYPERDRIVE_MODEL_PATH)

    hyperd_env = get_hyperd_environment()

    inference_config = InferenceConfig(
        environment=hyperd_env,
        source_directory='./source_dir',
        entry_script='./predict_hyperd.py',
        )

    deployment_config = AciWebservice.deploy_configuration(
        cpu_cores=0.5, memory_gb=1, auth_enabled=True
        )

    service = Model.deploy(ws, 'adult-hyperd-service', [model],
        inference_config, deployment_config, overwrite=True)
    service.wait_for_deployment(show_output=True)

    print(service.get_logs())


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


def create_hyperd_test_input(test_ds, row):
    test_df = test_ds.to_pandas_dataframe().iloc[row]
    test_label = int(test_df['income'])
    del test_df['income']
    test_input = cast_test_input(test_df)
    return test_input, test_label
    

def test_deployed_hyperd_model(test_ds, row):
    
    ws = get_workspace()
    
    service = Webservice(workspace=ws, name='adult-hyperd-service')
    scoring_uri = service.scoring_uri
    key, _ = service.get_keys()

    headers = {"Content-Type": "application/json"}
    headers['Authorization'] = f'Bearer {key}'
    test_input, test_label = create_hyperd_test_input(test_ds, row)
    data = json.dumps(test_input)
    response = requests.post(scoring_uri, data=data, headers=headers)
    
    print('label:', test_label, ', prediction:', int(response.json()))
 
    
def run_automl():
    
    from azureml.train.automl import AutoMLConfig
    from azureml.widgets import RunDetails

    ws = get_workspace()
    
    exp = Experiment(workspace=ws, name='adult-automl')
    run = exp.start_logging()
    
    compute_cluster = get_compute_cluster()
    
    train_ds, _ = get_data()
    
    automl_config = AutoMLConfig(
        experiment_timeout_minutes=30,
        task='classification',
        primary_metric='accuracy',
        training_data=train_ds,
        label_column_name='income',
        n_cross_validations=5,
        compute_target=compute_cluster)
    
    automl_run = exp.submit(automl_config)
    RunDetails(automl_run).show()
    automl_run.wait_for_completion(show_output=True)
    
    best_run, best_model = automl_run.get_output()
    best_run_metrics = best_run.get_metrics()

    print('Best run ID:', best_run.id)
    print('Accuracy:', best_run_metrics['accuracy'])

    joblib.dump(best_model, AUTOML_MODEL_PATH)
    

def test_local_automl_model(test_ds):
    model = joblib.load(AUTOML_MODEL_PATH)
    print(model)
    test_df = test_ds.to_pandas_dataframe()
    X_test = test_df.drop(['income'], axis=1)
    y_test = test_df['income']
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)
    return accuracy


def register_and_deploy_automl_model():
    
    ws = get_workspace()
    
    model = Model.register(ws, model_name='adult-automl-model',
        model_path=AUTOML_MODEL_PATH)

    automl_env = get_automl_environment()

    inference_config = InferenceConfig(
        environment=automl_env,
        source_directory='./source_dir',
        entry_script='./predict_automl.py',
        )

    deployment_config = AciWebservice.deploy_configuration(
        cpu_cores=0.5, memory_gb=1, auth_enabled=True
        )
    #deployment_config = LocalWebservice.deploy_configuration(port=6788)

    service = Model.deploy(ws, 'adult-automl-service', [model],
        inference_config, deployment_config, overwrite=True)
    service.wait_for_deployment(show_output=True)

    print(service.get_logs())
    
    return service

    
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


def create_automl_test_input(test_ds, row):
    test_df = test_ds.to_pandas_dataframe().iloc[row]
    test_label = test_df['income']
    del test_df['income']
    test_input = cast_automl_test_input(test_df)
    return test_input, test_label

    
def test_deployed_automl_model(test_ds, row, service=None):
    
    ws = get_workspace()
    
    if service is None:
        service = Webservice(workspace=ws, name='adult-automl-service')
    scoring_uri = service.scoring_uri
    key, _ = service.get_keys()

    headers = {"Content-Type": "application/json"}
    headers['Authorization'] = f'Bearer {key}'
    test_input, test_label = create_automl_test_input(test_ds, row)
    data = json.dumps(test_input)
    response = requests.post(scoring_uri, data=data, headers=headers)
    
    print('label:', test_label, ', prediction:', str(response.json()))
    

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--n_estimators', type=int, default=10, help='Number of estimators in the ensemble')
    parser.add_argument('--max_depth', type=int, default=5, help='Maximum depth of the trees involved')
    parser.add_argument('--max_features', type=int, default=14, help='Maximum number of features per tree')
    parser.add_argument('--min_samples_leaf', type=float, default=0.1, help='Minimum number of samples per leaf')

    args = parser.parse_args()
    
    run = Run.get_context()

    # Read parameters provided by HyperDrive run
    run.log('number of estimators', int(args.n_estimators))
    run.log('max depth', int(args.max_depth))
    run.log('max features', int(args.max_features))
    run.log('min samples leaf', float(args.min_samples_leaf))

    ws = run.experiment.workspace
    train_ds, _ = get_data(ws, suffix='hyperd')
    train_df = train_ds.to_pandas_dataframe()
    X_train = train_df.drop(['income'], axis=1).to_numpy()
    y_train = train_df['income'].to_numpy()
    
    clf = sklearn.ensemble.RandomForestClassifier(n_estimators=args.n_estimators,
        max_depth=args.max_depth, max_features=args.max_features,
        min_samples_leaf=args.min_samples_leaf, random_state=42)
    
    model_for_all_data = clf.fit(X_train, y_train)
    joblib.dump(model_for_all_data, './outputs/model.pkl')
    
    cv_results = cross_validate(clf, X_train, y_train, cv=5, scoring='accuracy', verbose=2)
    
    run.log("accuracy", float(cv_results['test_score'].mean()))


if __name__ == '__main__':
    main()

 