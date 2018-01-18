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
        rng = np.random.RandomState(seed=seed)
        n = len(dataset.X)
        indices = rng.permutation(n)
        splits = []

        groups = np.zeros(n, dtype=np.int)
        split_size = int(n/self.n_splits)
        n_bigger_splits = n % self.n_splits
        start_idx = 0
        for split_idx in range(self.n_splits):
            if n_bigger_splits > 0:
                end_idx = start_idx + split_size + 1
                n_bigger_splits -= 1
            else:
                end_idx = start_idx + split_size
            groups[start_idx:end_idx] = split_idx
            start_idx = end_idx

        for split_idx in range(self.n_splits):
            train_indices = indices[groups != split_idx]
            test_indices = indices[groups == split_idx]
            splits.append(Split(
                dataset[train_indices],
                dataset[test_indices]))

        return splits

def train_on_best_hyperparams(dataset, model_cls, hyperparams_list, splitter, score_function, seed):
    train_scores = []
    test_scores = []
    for n_split, split in enumerate(splitter.get_splits(dataset, seed=seed)):
        print("Evaluating split " + str(n_split+1), end='')
        split_train_scores = []
        split_test_scores = []
        for h in hyperparams_list:
            print('.', end='')
            model = model_cls(split.train.X, split.train.y, **h)
            split_train_scores.append(score_function(
                split.train.y,
                model.predict(split.train.X)))
            split_test_scores.append(score_function(
                split.test.y,
                model.predict(split.test.X)))
        print()
        train_scores.append(split_train_scores)
        test_scores.append(split_test_scores)
    train_scores = np.array(train_scores)
    test_scores = np.array(test_scores)
    best_h = hyperparams_list[np.argmax(np.average(test_scores, axis=0))]
    model = model_cls(dataset.X, dataset.y, **best_h)
    return model, best_h, train_scores, test_scores

def double_split_evaluate(dataset, model_cls, hyperparams_list, major_splitter, minor_splitter, score_function, seed):
    all_inner_train_scores = []
    all_inner_test_scores = []
    train_scores = []
    test_scores = []
    best_hs = []
    for i, major_split in enumerate(major_splitter.get_splits(dataset, seed=seed)):
        print("Evaluating major split", str(i+1))
        model, best_h, inner_train_scores, inner_test_scores = train_on_best_hyperparams(
            major_split.train,
            model_cls,
            hyperparams_list,
            minor_splitter,
            score_function,
            seed)
        train_scores.append(score_function(
            major_split.train.y,
            model.predict(major_split.train.X)))
        test_scores.append(score_function(
            major_split.test.y,
            model.predict(major_split.test.X)))
        all_inner_train_scores.append(inner_train_scores)
        all_inner_test_scores.append(inner_test_scores)
        best_hs.append(best_h)
    summary = {
        "best_hyperparams": best_hs,
        "train_scores": np.array(train_scores),
        "test_scores": np.array(test_scores),
        "estimated_score": np.average(np.array(test_scores)),
        "inner_train_scores": np.array(all_inner_train_scores),
        "inner_test_scores": np.array(all_inner_test_scores)}
    return summary
