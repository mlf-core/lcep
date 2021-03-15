import xgboost as xgb
import numpy as np

from rich import print
from dataclasses import dataclass


@dataclass
class Dataset:
    X: np.ndarray
    y: list
    DM: xgb.DMatrix
    gene_names: list
    sample_names: list


def load_train_test_data(train_data, test_data):
    X_train, y_train, train_gene_names, train_sample_names = parse_tpm_table(train_data)
    X_test, y_test, test_gene_names, test_sample_names_test = parse_tpm_table(test_data)

    print(f'[bold blue]Number of total samples: {str(len(X_train))}')
    print(f'[bold blue]Number of cancer samples: {str(len([x for x in y_train if x == 1]))}')
    print(f'[bold blue]Number of normal samples: {str(len([x for x in y_train if x == 0]))}')

    # Convert to Numpy Arrays
    X_train_np = np.array(X_train)
    X_test_np = np.array(X_test)

    # Convert from Numpy Arrays to XGBoost Data Matrices
    dtrain = xgb.DMatrix(X_train_np, label=y_train)
    dtest = xgb.DMatrix(X_test_np, label=y_test)

    training_data = Dataset(X_train_np, y_train, dtrain, train_gene_names, train_sample_names)
    test_data = Dataset(X_test, y_test, dtest, test_gene_names, test_sample_names_test)

    return training_data, test_data


def parse_tpm_table(input):
    X_train = []
    y_train = []
    gene_names = []
    sample_names = []
    with open(input, "r") as file:
        all_runs_info = next(file).split("\n")[0].split("\t")[2:]
        for run_info in all_runs_info:
            split_info = run_info.split("_")
            y_train.append(int(split_info[0]))
            sample_names.append(split_info[1])
        for line in file:
            splitted = line.split("\n")[0].split("\t")
            X_train.append([float(x) for x in splitted[2:]])
            gene_names.append(splitted[:2])

    X_train = [list(i) for i in zip(*X_train)]

    return X_train, y_train, gene_names, sample_names
