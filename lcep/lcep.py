from argparse import ArgumentParser
import xgboost as xgb
import mlflow
import mlflow.xgboost
import time
import numpy as np
import GPUtil
from rich import traceback, print

from mlf_core.mlf_core import MLFCore
from data_loading.data_loader import load_train_test_data
from evaluation.evaluation import calculate_log_metrics


def start_training():
    parser = ArgumentParser(description='XGBoost Example')
    parser.add_argument(
        '--max_epochs',
        type=int,
        default=25,
        help='Number of epochs to train',
    )
    parser.add_argument(
        '--general-seed',
        type=int,
        default=0,
        help='General Python, Python random and Numpy seed.',
    )
    parser.add_argument(
        '--xgboost-seed',
        type=int,
        default=0,
        help='XGBoost specific random seed.',
    )
    parser.add_argument(
        '--cuda',
        type=bool,
        default=True,
        help='Enable or disable CUDA support.',
    )
    parser.add_argument(
        '--single-precision-histogram',
        type=bool,
        default=True,
        help='Enable or disable single precision histogram calculation.',
    )
    parser.add_argument(
        '--training-data',
        type=str,
        help='Path to the training data',
    )
    parser.add_argument(
        '--test-data',
        type=str,
        help='Path to the test data',
    )
    avail_gpus = GPUtil.getGPUs()
    args = parser.parse_args()
    dict_args = vars(args)
    use_cuda = True if dict_args['cuda'] and len(avail_gpus) > 0 else False
    if use_cuda:
        print(f'[bold blue]Using {len(avail_gpus)} GPUs!')
    else:
        print('[bold blue]No GPUs detected. Running on the CPU')

    with mlflow.start_run():
        # Enable the logging of all parameters, metrics and models to mlflow
        mlflow.autolog(1)

        # Log hardware and software
        MLFCore.log_sys_intel_conda_env()

        # Fetch and prepare data
        training_data, test_data = load_train_test_data(dict_args['training_data'], dict_args['test_data'])

        # Enable input data logging
        # MLFCore.log_input_data('data/')

        # Set XGBoost parameters
        param = {'objective': 'binary:logistic',
                 'single_precision_histogram': True if dict_args['single_precision_histogram'] == 'True' else False,
                 'subsample': 0.7,
                 'colsample_bytree': 0.6,
                 'learning_rate': 0.2,
                 'max_depth': 3,
                 'min_child_weight': 1,
                 'eval_metric': 'logloss'}

        # Set random seeds
        MLFCore.set_general_random_seeds(dict_args["general_seed"])
        MLFCore.set_xgboost_random_seeds(dict_args["xgboost_seed"], param)

        # Set CPU or GPU as training device
        if use_cuda:
            param['tree_method'] = 'gpu_hist'
        else:
            param['tree_method'] = 'hist'

        # Train on the chosen device
        results = {}
        runtime = time.time()
        booster = xgb.train(param, training_data.DM, dict_args['max_epochs'], evals=[(test_data.DM, 'test')], evals_result=results)
        device = 'GPU' if use_cuda else 'CPU'
        if use_cuda:
            print(f'[bold green]{device} Run Time: {str(time.time() - runtime)} seconds')

        # Perform some predictions on the test data, evaluate and log them
        print('[bold blue]Performing predictions on test data.')
        test_predictions = np.round(booster.predict(test_data.DM))
        calculate_log_metrics(test_data.y, test_predictions)


if __name__ == '__main__':
    traceback.install()
    start_training()
