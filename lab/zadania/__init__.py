import os

m = __import__(os.environ['ML2018NAME'])

# zadanie 1
try:
    __import__(os.environ['ML2018NAME'] + '.zadanie1')
    gradient_descent = m.zadanie1.gradient_descent
    momentum = m.zadanie1.momentum
    nesterov = m.zadanie1.nesterov
    adam = m.zadanie1.adam
except ModuleNotFoundError:
    gradient_descent = None
    momentum = None
    nesterov = None
    adam = None

# zadanie 2
try:
    __import__(os.environ['ML2018NAME'] + '.zadanie2')
    bias_variance_estimation = m.zadanie2.bias_variance_estimation
except ModuleNotFoundError:
    bias_variance_estimation = None

# zadanie 3
try:
    __import__(os.environ['ML2018NAME'] + '.zadanie3')
    BayesianDensityEstimator = m.zadanie3.BayesianDensityEstimator
except ModuleNotFoundError:
    BayesianDensityEstimator = None

# zadanie 4
try:
    __import__(os.environ['ML2018NAME'] + '.zadanie4')
    BayesianLinearRegression = m.zadanie4.BayesianLinearRegression
except ModuleNotFoundError:
    BayesianLinearRegression = None
