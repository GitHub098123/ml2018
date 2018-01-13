import os
import shutil
import urllib.request

import numpy as np
import pandas as pd

CACHE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "cache")

class Dataset:

    def __init__(self, X, y, labels=None):
        self.X = X.copy()
        self.y = y.copy()
        self.labels = None if labels is None else labels.copy()

    def __getitem__(self, key):
        X = self.X[key]
        y = self.y[key]
        labels = None if self.labels is None else self.labels[key]
        return Dataset(X, y, labels)

    def shuffled_copy(self, seed):
        idx = np.random.RandomState(seed=seed).permutation(len(self.X))
        return self[idx]

class Split:

    def __init__(self, train_dataset, test_dataset):
        self.train = train_dataset
        self.test = test_dataset

class Splitter:

    def get_splits(self, dataset, seed):
        raise NotImplementedError()

class RandomSplitter(Splitter):

    def __init__(self, test_percentages):
        self.test_percentages = test_percentages

    def get_splits(self, dataset, seed=43):
        rng = np.random.RandomState(seed=seed)
        n = len(dataset.X)
        splits = []
        for p in self.test_percentages:
            n_test = int(p * n)
            indices = rng.permutation(n)
            train_indices = indices[n_test:]
            test_indices = indices[:n_test]
            splits.append(Split(
                dataset[train_indices],
                dataset[test_indices]))
        return splits

def download(url, filename):
    if not os.path.exists(CACHE_DIR):
        os.mkdir(CACHE_DIR)
    filename = os.path.join(CACHE_DIR, filename)
    if not os.path.exists(filename):
        with urllib.request.urlopen(url) as response, open(filename, 'wb') as f_out:
            shutil.copyfileobj(response, f_out)
    return filename

def _d_balanced_scale():
    filename = download(
        "http://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data", 
        "balance-scale.data")
    to_replace = {0: {
        "L": 0,
        "B": 1,
        "R": 2}}
    d = pd.read_csv(filename, header=None).replace(to_replace=to_replace)
    dataset = Dataset(X=d.loc[:, 1:].values, y=d[0].values, labels=None)
    dataset.dtypes = {
        "X": "categorical (ordered)",
        "y": "categorical (ordered)"}
    return dataset

def _d_banknote():
    filename = download(
        "http://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt", 
        "data_banknote_authentication.txt")
    d = pd.read_csv(filename, header=None)
    dataset = Dataset(X=d.loc[:, 0:3].values, y=d[4].values, labels=None)
    dataset.dtypes = {
        "X": "continuous",
        "y": "categorical"}
    return dataset

def _d_wine():
    filename = download(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", 
        "wine.data")
    d = pd.read_csv(filename, header=None)
    info = \
"""1) Alcohol (categorical)
2) Malic acid (continuous)
3) Ash (continuous)
4) Alcalinity of ash (continuous)  
5) Magnesium (continuous)
6) Total phenols (continuous)
7) Flavanoids (continuous)
8) Nonflavanoid phenols (continuous)
9) Proanthocyanins (continuous)
10)Color intensity (continuous)
11)Hue (continuous)
12)OD280/OD315 of diluted wines (continuous)
13)Proline (continuous)"""
    dataset = Dataset(X=d.loc[:, 0:12].values, y=d[13].values.astype(np.float), labels=None)
    dataset.info = info
    dataset.dtypes = {
        "X": "mixed",
        "y": "continuous"}
    return dataset

def _d_car():
    filename = download(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data", 
        "car.data")
    info = \
"""Attributes:
    buying: vhigh, high, med, low. (3, 2, 1, 0)
    maint: vhigh, high, med, low. (3, 2, 1, 0)
    doors: 2, 3, 4, 5more. (0, 1, 2, 3)
    persons: 2, 4, more. (0, 1, 2)
    lug_boot: small, med, big. (0, 1, 2)
    safety: low, med, high. (0, 1, 2)
Class Values:
    unacc, acc, good, vgood (0, 1, 2, 3)"""
    to_replace = {
        0: {
            "vhigh": 3,
            "high": 2,
            "med": 1,
            "low": 0},
        1: {
            "vhigh": 3,
            "high": 2,
            "med": 1,
            "low": 0},
        2: {
            "2": 0,
            "3": 1,
            "4": 2,
            "5more": 3},
        3: {
            "2": 0,
            "4": 1,
            "more": 2},
        4: {
            "small": 0,
            "med": 1,
            "big": 2},
        5: {
            "low": 0,
            "med": 1,
            "high": 2},
        6: {
            "unacc": 0,
            "acc": 1,
            "good": 2,
            "vgood": 3},
    }
    d = pd.read_csv(filename, header=None).replace(to_replace=to_replace)
    X = d.loc[:, 0:5].values
    y = d.loc[:, 6].values
    dataset = Dataset(X, y, labels=None)
    dataset.info = info
    dataset.dtypes = {
        "X": "categorical (ordered)",
        "y": "categorical (ordered)"}
    return dataset

def _d_iris():
    filename = download(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", 
        "iris.data")
    info = \
"""Attributes:
1. sepal length in cm
2. sepal width in cm
3. petal length in cm
4. petal width in cm
Class:
-- Iris Setosa (0)
-- Iris Versicolour (1)
-- Iris Virginica (2)"""
    to_replace = {
        4: {
            "Iris-setosa": 0,
            "Iris-versicolor": 1,
            "Iris-virginica": 2}
    }
    d = pd.read_csv(filename, header=None).replace(to_replace=to_replace)
    X = d.loc[:, 0:3].values
    y = d.loc[:, 4].values
    dataset = Dataset(X, y, labels=None)
    dataset.info = info
    dataset.dtypes = {
        "X": "continuous",
        "y": "categorical"}
    return dataset

_DATASETS = {
    "balance-scale": _d_balanced_scale,
    "banknote": _d_banknote,
    "car": _d_car,
    "iris": _d_iris,
    "wine": _d_wine,}

def list_datasets():
    return list(_DATASETS.keys())

def get_dataset(name):
    try:
        return _DATASETS[name]()
    except KeyError:
        raise ValueError("Dataset '" + name + "' not found.")

def train_test_split(dataset, test_percentage=.3, seed=43):
    return RandomSplitter(test_percentages=[test_percentage]).get_splits(dataset, seed=seed)[0]
