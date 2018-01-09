import numpy as np
from scipy.special import expit as sigmoid

from zadania import adam

# można zmodyfikować tę funkcję, użyć innego optimizera itp.
def find_gamma(initial_gamma, loss, n_steps):
    initial_gamma = np.array([initial_gamma])
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
    return - (- y_true * (1 - sigmoid(logits)) + (1 - y_true) * sigmoid(logits))

class GammaLoss:

    def __init__(self, y_true, logits, r):
        self.y_true = y_true
        self.logits = logits
        self.r = r

    def taylor(self, gamma):
        dummy_value = 0. # won't be used, const is faster
        return (
            dummy_value,
            - np.average(
                self.y_true * (1 - sigmoid(self.logits + gamma[0] * self.r)) * self.r - \
                (1 - self.y_true) * (sigmoid(self.logits + gamma[0] * self.r)) * self.r))

class GradientBoostingClassifier:

    def __init__(self, X, y, n_models, model_cls, train_fraction=.7, initial_gamma=1., gamma_n_steps=200, seed=43):

        self.models = []
        self.gammas = []

        rng = np.random.RandomState(seed=43)
        for step in range(n_models):
            indices = rng.permutation(np.arange(len(X)))[:int(train_fraction * len(X))]
            X_train = X[indices]
            y_train = y[indices]
            current_logits = self.predict_logits(X_train, step=step)
            r_true = binary_crossentropy_pseudo_residuals(y_train, current_logits)
            model = model_cls(X_train, r_true)
            r = model.predict(X_train)
            gamma = find_gamma(
                initial_gamma,
                GammaLoss(y_train, current_logits, r),
                n_steps=gamma_n_steps)
            self.models.append(model)
            self.gammas.append(gamma)

    def predict_logits(self, X, step=None):
        if step is None:
            step = len(self.models)
        predictions = np.zeros(len(X), dtype=np.float)
        for i in range(step):
            predictions += self.gammas[i] * self.models[i].predict(X)
        return predictions

    def predict(self, X, step=None):
        return (self.predict_logits(X, step) >= 0.).astype(np.int)
