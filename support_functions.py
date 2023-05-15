from sklearn.metrics import confusion_matrix
from typing import Tuple, List

import scipy.io as io
import numpy as np
import os


def delete_if_exist(data_path: str):
    if os.path.isfile(data_path):
        os.remove(data_path)


def to_file(file_name: str, array: np.ndarray) -> None:
    result_path = 'results'
    np.savetxt(fname=f"{result_path}/{file_name}", X=array, delimiter=",")


def mat_data_loader(data_path: str) -> dict:
    """
    Function for loading files with mat extension as dict.

    Parameters
    ----------
    data_path : str
        Path to the mat file.

    Returns
    -------
    data : np.ndarray
        Dictionary with data.
    """
    data = io.loadmat(data_path)
    return data


def unpack_data_mat_file(data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Unpack training and testing datasets and labels from mat file.

    Parameters
    ----------
    data_path : str
        Path to the mat file.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
    """
    data = mat_data_loader(data_path)
    train_data = data['data']
    test_data = data['datat']

    X_train = train_data[:, :-1]
    X_test = test_data[:, :-1]
    y_train = train_data[:, -1]
    y_test = test_data[:, -1]

    y_train[y_train == -1] = 0
    y_test[y_test == -1] = 0

    return X_train, X_test, y_train, y_test


def files_from_path(files_path: str, extension: str='.mat') -> list:
    """
    Create list of files from specific path with extension given.

    Parameters
    ----------
    files_path : basestring
    extension : str='.mat'

    Returns
    -------
    files_list : list
    """
    files_list = []
    for file in os.listdir(files_path):
        if file.endswith(extension):
            files_list.append(file)
    return files_list


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray, metrics_lst: list = None) -> dict:
    """
    Calculate metrics from metrics_lst. Used instead of
    Parameters
    ----------
    y_true : np.ndarray
    y_pred : np.ndarray
    metrics_lst : list

    Returns
    -------
    metrics : dict
        Dictionary with key-score metrics pair.
    """
    if metrics_lst is None:
        metrics_lst = ['accuracy', 'precision', 'recall', 'f1-score']

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics_dict = {}
    if 'accuracy' in metrics_lst:
        metrics_dict['acc'] = (tp+tn)/(tp+tn+fp+fn)
    if 'precision' in metrics_lst:
        metrics_dict['precision'] = tp/(tp+fp)
    if 'recall' in metrics_lst:
        metrics_dict['recall'] = tp/(tp+fn)
    if 'f1-score' in metrics_lst:
        metrics_dict['f1-score'] = fp/(fp+(fp+fn)/2)

    return metrics_dict
