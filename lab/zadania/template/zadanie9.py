import numpy as np
from scipy.special import expit as sigmoid

from zadania import adam

# można zmodyfikować tę funkcję, użyć innego optimizera itp.
def find_gamma(initial_gamma, loss, n_steps):
    initial_gamma = np.array([initial_gamma]) # GammaLoss oczekuje tablicy
    g = adam(
        f=loss,
        starting_point=initial_gamma,
        learning_rate=.1,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8)
    gamma = initial_gamma
    for _ in range(n_steps):
        gamma, _, _ = next(g)
    return gamma[0]

def binary_crossentropy_pseudo_residuals(y_true, logits):
    ...

class GammaLoss:

    def __init__(self, y_true, logits, r):
        self.y_true = y_true
        self.logits = logits
        self.r = r

    def taylor(self, gamma):
        ...

class GradientBoostingClassifier:

    def __init__(self, X, y, n_models, model_cls, train_fraction=.7, initial_gamma=1., gamma_n_steps=200, seed=43):

        self.models = []
        self.gammas = []

        ...

    def predict_logits(self, X, step=None):
        if step is None:
            step = len(self.models)
        ...

    def predict(self, X, step=None):
        return (self.predict_logits(X, step) >= 0.).astype(np.int)
