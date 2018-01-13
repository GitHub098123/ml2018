import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split


# accuracy_score

def mse_score(y_true, y_pred):
    return - mean_squared_error(y_true, y_pred)


def max_probability_prediction(y):
    return np.argmax(np.bincount(y))

def average_prediction(y):
    return np.average(y)


def get_mid_thresholds(X_column):
    a = np.unique(X_column)
    return .5 * (a[:-1] + a[1:])

def get_left_thresholds(X_column):
    return np.unique(X_column)[:-1]


def get_all_features(X):
    return range(X.shape[1])

def get_bootstrap_features_factory(seed):
    rng = np.random.RandomState(seed=seed)
    def get_bootstrap_features(X):
        return np.unique(rng.randint(X.shape[1], size=X.shape[1]))
    return get_bootstrap_features

# get_bootstrap_features = get_bootstrap_features_factory(seed=43)

def get_fraction_of_features_factory(fraction, seed):
    assert fraction > 0.
    assert fraction <= 1.
    rng = np.random.RandomState(seed=seed)
    def get_fraction_of_features(X):
        n_features = int(np.round(X.shape[1] * fraction))
        if n_features == 0:
            n_features = 1
        return sorted(rng.choice(X.shape[1], size=n_features, replace=False))
    return get_fraction_of_features

# get_fraction_of_features = get_fraction_of_features_factory(fraction=.5, seed=43)



def entropy_index(X, y, groups):

    def _entropy(y):
        if len(y) == 0:
            return 0.
        counts = np.bincount(y)
        p = counts / counts.sum()
        return np.sum(-np.multiply(p[p>0], np.log(p[p>0])))

    return np.sum([np.sum(groups==g) * _entropy(y[groups==g]) for g in np.unique(groups)]) / len(groups)

def variance_index(X, y, groups):

    # Gain defined as MSE loss reduction

    # Other possibilities:
    # * https://en.wikipedia.org/wiki/Decision_tree_learning#Variance_reduction
    # * http://www.saedsayad.com/decision_tree_reg.htm

    def _sum_of_squares(y):
        return np.sum(np.square(y-np.average(y)))
    return np.sum([_sum_of_squares(y[groups==g]) for g in np.unique(groups)])


class NodeSplitterException(Exception):
    pass

class NodeThresholdSplitter:

    @staticmethod
    def get_thresholds(X_column):
        raise NotImplementedError()

    @staticmethod
    def get_features(X):
        raise NotImplementedError()

    @staticmethod
    def split_index(X, y, groups):
        raise NotImplementedError()

    def __init__(self, X, y):

        best_split_index_value = self.split_index(X, y, np.zeros(len(y), dtype=int))
        best_feature_idx = None
        best_threshold = None

        groups = np.zeros(len(y), dtype=np.int)
        for feature_idx in self.get_features(X):
            for threshold in self.get_thresholds(X[:, feature_idx]):
                groups = (X[:, feature_idx]>threshold).astype(np.int)
                split_index_value = self.split_index(X, y, groups)
                if split_index_value < best_split_index_value:
                    best_split_index_value = split_index_value
                    best_feature_idx = feature_idx
                    best_threshold = threshold

        if best_feature_idx is None:
            raise NodeSplitterException()

        self.n_children = 2
        self.feature_idx = best_feature_idx
        self.threshold = best_threshold

    def predict(self, X):
        if len(X.shape) == 2:
            return (X[:, self.feature_idx] > self.threshold).astype(np.int)
        elif len(X.shape) == 1:
            return self.predict(np.array([X]))[0]
        else:
            raise ValueError()

    def split(self, X, y):
        indices = self.predict(X)
        return [(X[indices==child_idx], y[indices==child_idx]) for child_idx in range(self.n_children)]

    def __str__(self):
        return ' '.join([
            "f:",
            str(self.feature_idx),
            "thr:",
            str(self.threshold),
            ])


class Node:

    min_size = 2
    max_depth = None

    @staticmethod
    def make_prediction(y):
        raise NotImplementedError()

    @staticmethod
    def make_node_splitter(X, y):
        raise NotImplementedError()

    def __init__(self, X, y, depth):
        self.depth = depth
        self.prediction = self.make_prediction(y)
        self.node_splitter = None
        self.children = None
        self.flag_early_prediction = False # useful for pruning

        if \
                (len(X) < self.min_size) or \
                (self.max_depth is not None and self.depth >= self.max_depth) or \
                (len(np.unique(y)) == 1):
            self.to_leaf()
        else:
            try:
                self.node_splitter = self.make_node_splitter(X, y)
                self.children = tuple([self.__class__(X, y, self.depth + 1) for X, y in self.node_splitter.split(X, y)])
            except NodeSplitterException:
                self.to_leaf()

    def to_leaf(self):
        self.node_splitter = None
        self.children = ()

    def predict(self, x):
        if self.flag_early_prediction is True or self.children == ():
            return self.prediction
        else:
            return self.children[self.node_splitter.predict(x)].predict(x)

    def __str__(self):
        result = str(self.node_splitter) + ' ' if self.node_splitter is not None else ""
        result += "pred: " + str(self.prediction)
        return result


class DecisionTree:

    def __init__(self, X, y, min_size=2, max_depth=None, prune=False, seed=43):

        if prune is True:
            X_train, X_valid, y_train, y_valid = train_test_split(
                X, y, test_size=0.2, random_state=seed)
        else:
            X_train = X
            y_train = y

        # BINARY TREE CLASSIFIER, CONTINUOUS FEATURES, ENTROPY INDEX
        class _NodeThresholdSplitter(NodeThresholdSplitter):
            get_thresholds = staticmethod(get_left_thresholds)
            get_features = staticmethod(get_all_features)
            split_index = staticmethod(entropy_index)
        class _Node(Node):
            make_prediction = staticmethod(max_probability_prediction)
            make_node_splitter = staticmethod(_NodeThresholdSplitter)
        _Node.min_size = min_size
        _Node.max_depth = max_depth
        fn_score = accuracy_score

        """
        # BINARY TREE CLASSIFIER, CONTINUOUS FEATURES, ENTROPY INDEX, BOOTSTRAP FEATURES
        class _NodeThresholdSplitter(NodeThresholdSplitter):
            get_thresholds = staticmethod(get_left_thresholds)
            split_index = staticmethod(entropy_index)
        _NodeThresholdSplitter.get_features = staticmethod(get_bootstrap_features_factory(seed=seed))
        class _Node(Node):
            make_prediction = staticmethod(max_probability_prediction)
            make_node_splitter = staticmethod(_NodeThresholdSplitter)
        _Node.min_size = min_size
        _Node.max_depth = max_depth
        fn_score = accuracy_score
        """

        """
        # BINARY TREE REGRESSOR, CONTINUOUS FEATURES, VARIANCE INDEX
        class _NodeThresholdSplitter(NodeThresholdSplitter):
            get_thresholds = staticmethod(get_left_thresholds)
            split_index = staticmethod(variance_index)
            get_features = staticmethod(get_all_features)
        class _Node(Node):
            make_prediction = staticmethod(average_prediction)
            make_node_splitter = staticmethod(_NodeThresholdSplitter)
        _Node.min_size = min_size
        _Node.max_depth = max_depth
        fn_score = mse_score
        """

        self.root = _Node(X_train, y_train, depth=0)

        if prune is True:
            self.prune(X_valid, y_valid, fn_score)

    def prune(self, X_valid, y_valid, fn_score):
        def apply_dfs_leaf_first(node, fn):
            for child in node.children:
                apply_dfs_leaf_first(child, fn)
            fn(node)
        def prune_node(node):
            score = fn_score(y_valid, self.predict(X_valid))
            node.flag_early_prediction = True
            pruned_score = fn_score(y_valid, self.predict(X_valid))
            node.flag_early_prediction = False
            if pruned_score > score:
                node.to_leaf()
        apply_dfs_leaf_first(self.root, prune_node)

    def predict(self, X):
        return np.array([self.root.predict(x) for x in X], dtype=np.int)

    def __str__(self):
        def print_node(node):
            return node.depth * '-' + ' ' + str(node)
        def f(node):
            result = [print_node(node)]
            for child in node.children:
                result += f(child)
            return result
        return '\n'.join(f(self.root))
