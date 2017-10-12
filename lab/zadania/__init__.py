import os

m = __import__(os.environ['ML2018NAME'])
__import__(os.environ['ML2018NAME'] + '.zadanie1')

# zadanie 1
gradient_descent = m.zadanie1.gradient_descent
momentum = m.zadanie1.momentum
nesterov = m.zadanie1.nesterov
adam = m.zadanie1.adam
