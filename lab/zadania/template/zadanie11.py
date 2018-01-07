import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class SimpleLogisticAgent:

    def __init__(self, n_features, learning_rate=.1, seed=43):
        self.w = np.random.RandomState(seed=2*seed).normal(size=(n_features,))
        self.b = 0.
        self.learning_rate = learning_rate
        self.rng = np.random.RandomState(seed=seed)

    def predict_proba(self, X):
        return sigmoid(np.dot(X, self.w) + self.b)

    def predict(self, X):
        return (self.predict_proba(X) >= .5).astype(np.uint8)

    def actions(self, X):
        ...

    def update(self, X, actions, rewards):
        ...
