
import numpy as np

def gradient_descent(f, starting_point, learning_rate):
    theta = starting_point.copy()
    yield (theta,) + tuple(f.taylor(theta))
    ...
    while True:
        ...
        yield (theta,) + tuple(f.taylor(theta))

def momentum(f, starting_point, learning_rate, gamma):
    theta = starting_point.copy()
    yield (theta,) + tuple(f.taylor(theta))
    ...
    while True:
        ...
        yield (theta,) + tuple(f.taylor(theta))

def nesterov(f, starting_point, learning_rate, gamma):
    theta = starting_point.copy()
    yield (theta,) + tuple(f.taylor(theta))
    ...
    while True:
        ...
        yield (theta,) + tuple(f.taylor(theta))

def adam(f, starting_point, learning_rate, beta1, beta2, epsilon):
    theta = starting_point.copy()
    yield (theta,) + tuple(f.taylor(theta))
    ...
    while True:
        ...
        yield (theta,) + tuple(f.taylor(theta))
