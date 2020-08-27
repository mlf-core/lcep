import xgboost as xgb
import numpy as np

from rich import print


def load_train_test_data(train_data, test_data):
    X_train, y_train, gene_names, sample_names_train = parse_tpm_table(train_data)
    X_test, y_test, gene_names, sample_names_test = parse_tpm_table(test_data)

    print(f'[bold blue]Number of total samples: {str(len(X_train))}')
    print(f'[bold blue]Number of cancer samples: {str(len([x for x in y_train if x == 1]))}')
    print(f'[bold blue]Number of normal samples: {str(len([x for x in y_train if x == 0]))}')

    X_train = np.array(X_train)
    X_test = np.array(X_test)

    # Convert input data from numpy to XGBoost format
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    return dtrain, dtest, gene_names, sample_names_train, sample_names_test


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
