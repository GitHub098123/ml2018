import numpy as np
import tqdm

class D1:
    # D1 - tak oznaczamy funkcje, które są różniczkowalne
    def __call__(self, x):
        return self.taylor(x)[0]

class Bilinear(D1):

    def taylor(self, w_x):
        assert len(w_x.shape) == 1 # input to wektor
        w, x = w_x[:int(len(w_x)/2)], w_x[int(len(w_x)/2):] # rozbijamy w_x na w oraz x
        # wartość to iloczyn skalarny <x,w>
        # d<x,w>/dx = w; d<x,w>/dw = x; sklejamy 2 gradienty
        return \
            np.dot(x, w), \
            np.hstack((x, w)).reshape(1,-1)

class BilinearOnX(D1):

    def __init__(self, x):
        self.x = x

    def taylor(self, w):
        assert w.shape == self.x.shape
        w_x = np.hstack((w, self.x))
        output, grad_w_x = Bilinear().taylor(w_x)
        grad_w = grad_w_x[:, :len(w)]
        return output, grad_w

class HalfSquaredEuclidDistance(D1):

    def taylor(self, y1_y2):
        # input to wektor, czyli jednowymiarowa macierz
        assert len(y1_y2.shape) == 1
        # obliczamy długość wektora y1
        half_len = int(len(y1_y2)/2)
        # rozbijamy y1_y2 na y1 oraz y2
        y1, y2 = y1_y2[:half_len], y1_y2[half_len:]
        # wartość funkcji: wzór znany jako koszt MSE, tutaj dla jednej pary (y1, y2) oraz bez znaku '-'
        # gradient: dMSE/dy1 = y1-y2; dMSE/dy2 = y2-y1; sklejamy 2 gradienty
        return \
            .5 * np.sum(np.square(y1 - y2)), \
            np.hstack(((y1 - y2), (y2 - y1))).reshape(1,-1)

class HalfSquaredEuclidDistanceOnY(D1):

    def __init__(self, y):
        self.y = y

    def taylor(self, y):
        assert y.shape == self.y.shape
        y1_y2 = np.hstack((y, self.y))
        output, grad_y1_y2 = HalfSquaredEuclidDistance().taylor(y1_y2)
        grad_y = grad_y1_y2[:, :len(y)]
        return output, grad_y

class LinearRegressionOnDataset(D1):

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def next_batch(self):
        return zip(self.X, self.Y)

    def taylor(self, w):
        assert len(w.shape) == 1
        values = []
        grads = []
        for x, y in self.next_batch():
            f1 = BilinearOnX(x)
            f2 = HalfSquaredEuclidDistanceOnY(y)
            f1_on_w, d_f1_d_w = f1.taylor(w)
            f1_on_w = np.array([f1_on_w])
            f2_on_f1_on_w, d_f2_d_f1_on_w = f2.taylor(f1_on_w)

            f2_f1_on_w = f2_on_f1_on_w
            d_f2_f1_d_w = np.dot(d_f2_d_f1_on_w, d_f1_d_w)

            values.append(f2_f1_on_w)
            grads.append(d_f2_f1_d_w)
            
        return \
            sum(values) / len(values), \
            sum(grads) / len(grads)

class LinearRegressionModel:

    def __init__(self, X, Y, optimizer, optimizer_kwargs, n_steps, seed, weights=None, progress_bar=True):
        if weights is None:
            rng_start = np.random.RandomState(seed=seed)
            starting_point = rng_start.normal(size=(X.shape[1],))
            starting_point[0] = 0
        else:
            starting_point = weights.copy()
        opt = optimizer(
            f=LinearRegressionOnDataset(X, Y),
            starting_point=starting_point,
            **optimizer_kwargs)
        next(opt) # starting_point
        _range = \
            tqdm.tqdm(range(n_steps)) if progress_bar else range(n_steps)
        for _ in _range:
            self.w, _, _ = next(opt)

    @property
    def weights(self):
        return self.w.copy()

    def predict(self, X):
        assert X.shape[1] == self.w.shape[0]
        ys = []
        for x in X:
            ys.append(Bilinear().taylor(np.hstack((self.w, x)))[0])
        return np.array(ys).reshape(-1,1)
