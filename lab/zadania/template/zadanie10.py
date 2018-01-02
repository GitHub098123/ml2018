from itertools import product

import numpy as np

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

class RandomLabelSplitter(Splitter):

    def __init__(self, test_percentages):
        self.test_percentages = test_percentages

    def get_splits(self, dataset, seed=43):
        assert dataset.labels is not None, "'dataset.labels' cannot be None"
        rng = np.random.RandomState(seed=seed)
        unique_labels = np.unique(dataset.labels)
        n = len(unique_labels)
        splits = []
        for p in self.test_percentages:
            n_test = int(p * n)
            indices = rng.permutation(n)
            train_mask = np.isin(dataset.labels, unique_labels[indices[n_test:]])
            test_mask = np.isin(dataset.labels, unique_labels[indices[:n_test]])        
            splits.append(Split(
                dataset[train_mask],
                dataset[test_mask]))
        return splits

class TimeSplitter(Splitter):

    def __init__(self, test_percentages):
        self.test_percentages = test_percentages

    def get_splits(self, dataset, seed=43):
        assert dataset.labels is not None, "'dataset.labels' cannot be None"
        rng = np.random.RandomState(seed=seed)
        n = len(dataset.X)
        splits = []
        indices = np.argsort(dataset.labels)
        for p in self.test_percentages:
            n_test = int(p * n)
            # bez ".copy()" rng.shuffle robi brzydkie rzeczy tablicy "indices"
            train_indices = indices[:n-n_test].copy()
            test_indices = indices[n-n_test:].copy()
            rng.shuffle(train_indices)
            rng.shuffle(test_indices)
            splits.append(Split(
                dataset[train_indices],
                dataset[test_indices]))
        return splits

def random_grid_search(grid, n=None, seed=43):
    keys, values = zip(*list(grid.items()))
    prod = list(product(*values))
    np.random.RandomState(seed=seed).shuffle(prod)
    if n is None:
        n = len(prod)
    return [dict(zip(keys, v)) for v in prod[:n]]

# TU ZACZYNA SIĘ ZADANIE 10

class CVSplitter(Splitter):

    def __init__(self, n_splits):
        self.n_splits = n_splits

    def get_splits(self, dataset, seed=43):
        ...

def train_on_best_hyperparams(dataset, model_cls, hyperparams_list, splitter, score_function, seed):
    ...

def double_split_evaluate(dataset, model_cls, hyperparams_list, major_splitter, minor_splitter, score_function, seed):
    ...
