import numpy as np

from zadania import adam

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LogisticRegressionCost:

    def __init__(self, X, y, reg_lambda):
        self.X = X
        self.y = y
        self.reg_lambda = reg_lambda

    def taylor(self, x):
        # x == (w, b)
        w = x[:-1]
        b = x[-1]
        y_pred = sigmoid(np.dot(self.X, w) + b) 
        dw = (np.dot(self.X.T, (y_pred - self.y)) + self.reg_lambda * w) / len(self.X)
        db = np.average(y_pred - self.y) + self.reg_lambda * b / len(self.X)
        dx = np.concatenate((dw, np.array([db])))
        return (
            0., # dummy value, won't be needed
            dx)

class LogisticRegression:

    def __init__(self, X, y, reg_lambda=1., n_iter=100, tol=.0001, seed=43):
        w = np.random.RandomState(seed=seed).normal(size=(X.shape[1],))
        b = 0
        x = np.concatenate((w, np.array([b])))
        optimizer = adam(
            f=LogisticRegressionCost(X, y, reg_lambda),
            starting_point=x,
            learning_rate=.1,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8)
        MAX_REPEATS = 43
        for _ in range(MAX_REPEATS):
            for _, (x, _, grad) in zip(range(n_iter), optimizer):
                pass
            if np.average(np.abs(grad)) < tol:
                break
        self.w = x[:-1]
        self.b = x[-1]
    
    def predict_proba(self, X):
        y_pred = sigmoid(np.dot(X, self.w) + self.b)
        return np.stack((1 - y_pred, y_pred), axis=1)

    def predict(self, X):
        return (self.predict_proba(X)[:,1].reshape(-1,1) >= .5).astype(np.uint8)
