import numpy as np

class BayesianLinearRegression:

    def __init__(self, noise_distr, params, prior):
        self.params = params.copy()
        self._posterior = prior.copy()
        self.noise_distr = noise_distr

    def _delta(self, x, y, params):
        assert params.shape == (2,)
        y_pred = params[0] + params[1] * x
        return y - y_pred

    @property
    def posterior(self):
        return self._posterior.copy()

    def normalized_likelihood(self, obs_x, obs_y):
        return np.array([
            np.prod(self.noise_distr.pdf(self._delta(obs_x, obs_y, params))) \
                for params in self.params])

    def conditional_ppd(self, x, y):
        l = np.array([self.noise_distr.pdf(self._delta(x, y, params)).ravel() for params in self.params])
        l = np.multiply(self._posterior.reshape((-1,1)), l)
        l = l.sum(axis=0)
        return l

    def observe(self, obs_x, obs_y):
        self._posterior = np.multiply(self._posterior, self.normalized_likelihood(obs_x, obs_y))
        self._posterior /= self._posterior.sum()
