from classifiers.clonax import CLONAX
from support_functions import calc_metrics, delete_if_exist

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ParameterGrid

import pandas as pd
import numpy as np
import time
import json
import configparser

config = configparser.ConfigParser()


def main():

    config.read('config.ini')

    metrics = json.loads(config.get('metrics', 'metrics_list'))
    datasets_path = json.loads(config.get('files', 'dataset'))
    output_path = json.loads(config.get('files', 'output_file'))

    delete_if_exist(output_path)
    params_grid = generate_params_grid()

    results_lst = []

    dataset = pd.read_csv(datasets_path, index_col=0)
    dataset = dataset[(dataset != '?').all(axis=1)]

    X = np.array(dataset.iloc[:, 1:-1])
    y = np.array(dataset.iloc[:, -1])

    n_folds = config.getint('CV', 'n_folds')

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1)
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        for params in params_grid:
            start_time = time.time()
            y_pred = train_and_pred(X_train, X_test, y_train, params)
            print(f"--- {time.time() - start_time} seconds ---")

            score = calc_metrics(y_test, y_pred, metrics_lst=metrics)

            results_lst.append([datasets_path, *params, score['acc'], score['f1-score']])
    df_results = pd.DataFrame(results_lst)
    df_results.to_csv(output_path, mode='a', header=False, index=False)


def train_and_pred(X_train, X_test, y_train, params):
    clonax = CLONAX(generations=int(params[0]),
                    memory_size=int(params[1]),
                    remaining_ratio=float(params[2]),
                    replaceable_size_ratio=float(params[3]),
                    n_to_clone=int(params[4]),
                    n_best_clones=int(params[5]),
                    n_antigens_to_average=int(params[6]),
                    with_proportion=bool(params[7]))

    clonax = clonax.fit(X_train, y_train)
    y_pred = clonax.predict(X_test)

    del clonax
    return y_pred


def generate_params_grid():
    params_dict = {}
    for key, list_as_string in config.items('CLONAX_params'):
        params_dict[key] = json.loads(list_as_string)

    params_grid = list(ParameterGrid(params_dict))
    df = pd.DataFrame(params_grid)
    df = df[['generations', 'memory_size', 'remaining_ratio', 'replaceable_size_ratio',
             'n_to_clone', 'n_best_clones', 'n_antigens_to_average', 'with_proportion']]

    params_list = []
    for idx_row in range(df.shape[0]):
        params_list.append(df.iloc[idx_row, :].values.tolist())
    return np.array(params_list)


if __name__ == '__main__':
    main()
