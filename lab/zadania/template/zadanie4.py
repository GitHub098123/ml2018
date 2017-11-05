import numpy as np

class BayesianLinearRegression:

    def __init__(self, noise_distr, params, prior):
        ...

    @property
    def posterior(self):
        ...

    def normalized_likelihood(self, obs_x, obs_y):
        ...

    def conditional_ppd(self, x, y):
        ...

    def observe(self, obs_x, obs_y):
        ...
