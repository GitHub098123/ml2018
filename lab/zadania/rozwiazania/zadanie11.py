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
        return self.rng.binomial(1, self.predict_proba(X))

    def update(self, X, actions, rewards):
        y_pred = self.predict_proba(X)
        
        dW = np.zeros(X.shape)
        dW[actions == 1] = ((1 - y_pred).reshape((-1,1)) * X * rewards.reshape((-1,1)))[actions == 1]
        dW[actions == 0] = (- y_pred.reshape((-1,1)) * X * rewards.reshape((-1,1)))[actions == 0]
        dw = np.average(dW, axis=0)

        dB = np.zeros(actions.shape)
        dB[actions == 1] = ((1 - y_pred) * rewards)[actions == 1]
        dB[actions == 0] = (- y_pred * rewards)[actions == 0]
        db = np.average(dB, axis=0)

        self.w += self.learning_rate * dw
        self.b += self.learning_rate * db
