import numpy as np

class LogisticRegression:

    def __init__(self, X, y):
        ...

    @property
    def w(self):
        ...

    @property
    def b(self):
        ...

    def predict_proba(self, X):
        ...

    def predict(self, X):
        return (self.predict_proba(X)[:,1].reshape(-1,1) >= .5).astype(np.uint8)
