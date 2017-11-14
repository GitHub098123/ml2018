import numpy as np

class BayesianDensityEstimator:

    def __init__(self, distr_cls, params, prior):
        self.params = params.copy()
        self._posterior = prior.copy()
        self.distr_cls = distr_cls

    @property
    def posterior(self):
        return self._posterior.copy()

    def likelihood(self, obs):
        return np.array([
            np.prod(self.distr_cls(param).pdf(obs)) \
                for param in self.params])

    def ppd(self, x):
        l = np.array([self.distr_cls(param).pdf(x) for param in self.params])
        l = np.multiply(self._posterior.reshape((-1,1)), l)
        l = l.sum(axis=0)
        return l

    def observe(self, obs):
        self._posterior = np.multiply(self._posterior, self.likelihood(obs))
        self._posterior /= self._posterior.sum()
