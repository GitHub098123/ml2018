import os

m = __import__(os.environ['ML2018NAME'])
__import__(os.environ['ML2018NAME'] + '.zadanie1')
__import__(os.environ['ML2018NAME'] + '.zadanie2')

# zadanie 1
gradient_descent = m.zadanie1.gradient_descent
momentum = m.zadanie1.momentum
nesterov = m.zadanie1.nesterov
adam = m.zadanie1.adam

# zadanie 2
bias_variance_estimation = m.zadanie2.bias_variance_estimation
