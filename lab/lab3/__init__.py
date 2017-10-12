import numpy as np
from numpy.polynomial.polynomial import polyval2d

class ExpPoly2D:

    def __init__(self, cs, ds):
        self.cs = cs
        self.ds = ds

    def _output(self, x1, x2, c, d):
        return d * np.exp(-np.polynomial.polynomial.polyval2d(x1, x2, c)**2)

    def _d_dx1(self, x1, x2, c, d):
        return \
            d * np.exp(-polyval2d(x1, x2, c)**2) * \
            (-2.) * polyval2d(x1, x2, c) * \
            polyval2d(x1, x2, np.multiply(np.arange(c.shape[0]).reshape(-1,1), c)[1:,:])
            
    def _d_dx2(self, x1, x2, c, d):
        return \
            d * np.exp(-polyval2d(x1, x2, c)**2) * \
            (-2.) * polyval2d(x1, x2, c) * \
            polyval2d(x1, x2, np.multiply(np.arange(c.shape[1]).reshape(1,-1), c)[:,1:])
        
    def taylor(self, x):
        assert x.shape == (2,)
        x1, x2 = x
        return \
            np.sum([self._output(x1, x2, c, d) for c, d in zip(self.cs, self.ds)]), \
            np.array([[
                np.sum([self._d_dx1(x1, x2, c, d) for c, d in zip(self.cs, self.ds)]),
                np.sum([self._d_dx2(x1, x2, c, d) for c, d in zip(self.cs, self.ds)]),
            ]])

    def __call__(self, x):
        return self.taylor(x)[0]

def gradient_ascent(f, starting_point, learning_rate):
    theta = starting_point.copy()
    yield (theta,) + tuple(f.taylor(theta))
    while True:
        output, gradient = f.taylor(theta)
        yield theta, output, gradient
        theta += learning_rate * gradient.reshape(theta.shape)

def normalized_gradient_ascent(f, starting_point, learning_rate):
    theta = starting_point.copy()
    yield (theta,) + tuple(f.taylor(theta))
    while True:
        output, gradient = f.taylor(theta)
        yield theta, output, gradient
        theta += \
            learning_rate * \
            gradient.reshape(theta.shape) / \
            np.linalg.norm(gradient)
