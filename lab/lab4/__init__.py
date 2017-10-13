import numpy as np
from numpy import poly1d
from .LinearRegression import LinearRegressionModel

def polynomial_base(X, degree):
    assert len(X.shape) == 2
    assert X.shape[1] == 1
    return np.hstack([np.power(X, n) for n in range(degree)])

def fourier_base(X, degree):
    assert len(X.shape) == 2
    assert X.shape[1] == 1
    return np.hstack((
        np.hstack([np.cos(n*X) for n in range(degree+1)]),
        np.hstack([np.sin((n+1)*X) for n in range(degree)])))

class ToyDataGenerator:

    def __init__(self, seed, sigma, degree=6):
        assert sigma > 0., "'sigma' must be positive."
        assert degree > 0, "'degree' must be positive."
        self.rng = np.random.RandomState(seed=seed)
        self.degree = degree
        self.weights = self.rng.uniform(1,2,size=2*degree+1)
        self.sigma = sigma

    def sample(self, n=None):
        X, Y = self.noiseless_sample(n)
        Y += self.rng.normal(loc=0., scale=self.sigma, size=len(Y)).reshape(Y.shape)
        return X, Y

    def noiseless_sample(self, n):
        X = self.rng.uniform(-2, 2, size=n).reshape(-1,1)
        Y = np.dot(fourier_base(X, self.degree), self.weights).reshape(-1,1) / self.degree
        return X, Y
