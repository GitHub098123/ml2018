
import numpy as np

def gradient_descent(f, starting_point, learning_rate):
    theta = starting_point.copy()
    yield (theta,) + tuple(f.taylor(theta))
    while True:
        theta -= learning_rate * f.taylor(theta)[1].reshape(theta.shape)
        yield (theta,) + tuple(f.taylor(theta))

def momentum(f, starting_point, learning_rate, gamma):
    theta = starting_point.copy()
    yield (theta,) + tuple(f.taylor(theta))
    v = 0. * theta
    while True:
        v = gamma * v + learning_rate * f.taylor(theta)[1].reshape(theta.shape)
        theta -= v 
        yield (theta,) + tuple(f.taylor(theta))

def nesterov(f, starting_point, learning_rate, gamma):
    theta = starting_point.copy()
    yield (theta,) + tuple(f.taylor(theta))
    v = 0. * starting_point
    while True:
        x = theta - gamma * v
        v = gamma * v + learning_rate * f.taylor(x)[1].reshape(theta.shape)
        theta -= v
        yield (theta,) + tuple(f.taylor(theta))

def adam(f, starting_point, learning_rate, beta1, beta2, epsilon):
    theta = starting_point.copy()
    yield (theta,) + tuple(f.taylor(theta))
    m = 0. * theta
    v = 0. * theta
    t = 1
    while True:
        grad = f.taylor(theta)[1].reshape(theta.shape)
        m = beta1 * m + (1. - beta1) * grad
        v = beta2 * v + (1. - beta2) * np.square(grad)
        m_hat = m / (1. - beta1**t)
        v_hat = v / (1. - beta2**t)
        t += 1
        theta -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        yield (theta,) + tuple(f.taylor(theta))
