import numpy as np
import tqdm
from tqdm import tnrange

from lab4 import LinearRegressionModel, ToyDataGenerator, fourier_base
from .zadanie1 import adam

def bias_variance_estimation(
    generator_true_degree,
    generator_sigma,
    generator_seed,
    n_train_samples,
    max_degree,
    n_simulations_per_degree):

    n_test_samples = 3000
    optimizer = adam
    linear_regression_seed = 43
    linear_regression_n_steps = 100
    optimizer_kwargs = {
        "learning_rate": .1,
        "beta1": .9,
        "beta2": .999,
        "epsilon": 1e-8}

    ...
