import numpy as np

class BayesianDensityEstimator:

    def __init__(self, distr_cls, params, prior):
        ...

    @property
    def posterior(self):
        ...

    def likelihood(self, obs):
        ...

    def ppd(self, x):
        ...

    def observe(self, obs):
        ...
