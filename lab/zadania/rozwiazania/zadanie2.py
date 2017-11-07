
import numpy as np
import tqdm
from tqdm import tnrange

from lab4 import LinearRegressionModel, ToyDataGenerator, fourier_base
from .zadanie1 import adam

def bias_variance_estimation(
    generator_true_degree=6,
    generator_sigma=.3,
    generator_seed=437,
    n_train_samples=30,
    max_degree=10,
    n_simulations_per_degree=10):

    n_test_samples = 3000
    optimizer = adam
    linear_regression_seed = 43
    linear_regression_n_steps = 100
    optimizer_kwargs = {
        "learning_rate": .1,
        "beta1": .9,
        "beta2": .999,
        "epsilon": 1e-8}

    tdg = ToyDataGenerator(
        seed=generator_seed,
        sigma=generator_sigma,
        degree=generator_true_degree)

    noise = generator_sigma**2

    X_noiseless, Y_noiseless = tdg.noiseless_sample(n_test_samples)
    X_test, Y_test = tdg.sample(n_test_samples)

    degrees = []
    squared_biases = []
    variances = []
    average_mses = []

    for _d in tnrange(max_degree):
        degree = _d + 1
        Y_noiseless_preds = []
        mses = []
        for _ in tnrange(n_simulations_per_degree):
            X_train, Y_train = tdg.sample(n_train_samples)
            model = LinearRegressionModel(
                fourier_base(X_train, degree),
                Y_train,
                optimizer,
                optimizer_kwargs,
                n_steps=linear_regression_n_steps,
                seed=linear_regression_seed,
                progress_bar=False)
            Y_noiseless_preds.append(model.predict(
                fourier_base(X_noiseless, degree)))
            mses.append(
                np.average(np.square(
                    Y_test - model.predict(
                        fourier_base(X_test, degree)))))

        Y_noiseless_preds_mean = \
            sum(Y_noiseless_preds) / len(Y_noiseless_preds)
        bias_squared = np.average(np.square(
            Y_noiseless_preds_mean - Y_noiseless))
        variance = np.average([
            np.average(
                np.square(Y_noiseless_pred - Y_noiseless_preds_mean)) \
            for Y_noiseless_pred in Y_noiseless_preds])
        degrees.append(degree)
        squared_biases.append(bias_squared)
        variances.append(variance)
        average_mses.append(np.average(mses))

    return (
        np.array(degrees),
        np.array(squared_biases),
        np.array(variances),
        noise * np.ones_like(degrees),
        np.array(average_mses))
