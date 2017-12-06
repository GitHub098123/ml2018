import numpy as np
import scipy
from sklearn.feature_extraction.text import CountVectorizer

def unnormalized_logposteriors_to_logits(ulp):
    assert len(ulp.shape) == 2
    ulp -= np.max(ulp, axis=1).reshape(-1,1)
    ulp -= np.log(np.sum(np.exp(ulp), axis=1)).reshape(-1,1)
    return ulp

class TextMultinomialNaiveBayes:

    def __init__(self, X, y):
        n_classes = int(max(y) + 1)
        assert set(y) == set(np.arange(n_classes))

        self.cv = CountVectorizer(min_df=5)
        self.cv.fit(X)
        self.features = list(zip(*sorted(
            self.cv.vocabulary_.items(),
            key=lambda tup: tup[1])))[0]

        self.priors = np.bincount(y) / len(y)

        X = self.cv.transform(X)
        l = np.concatenate([
            np.array((X[y==i].sum(axis=0)) + 1) \
                for i in range(n_classes)], axis=0)
        self.likelihoods = l / l.sum(axis=1).reshape(-1,1)

        self.rng = np.random.RandomState(seed=43)

    def generate_sentence(self, nb_class, length):
        return ' '.join([self.features[i] for i in self.rng.choice(
            np.arange(len(self.features)),
            size=length,
            p=self.likelihoods[nb_class])])

    def predict_logits(self, X):
        return unnormalized_logposteriors_to_logits(
            self.cv.transform(X).dot(
                np.log(self.likelihoods).T) + \
                    np.log(self.priors).reshape(1,-1))

    def predict(self, X):
        return np.argmax(self.predict_logits(X), axis=1)

class TextBernoulliNaiveBayes:

    def __init__(self, X, y):
        n_classes = int(max(y) + 1)
        assert set(y) == set(np.arange(n_classes))

        self.cv = CountVectorizer(min_df=5)
        self.cv.fit(X)
        self.features = list(zip(*sorted(
            self.cv.vocabulary_.items(),
            key=lambda tup: tup[1])))[0]

        self.priors = np.bincount(y) / len(y)

        X = self.cv.transform(X)
        X.data[:] = 1

        l = np.concatenate([
            np.array((X[y==i].sum(axis=0)) + 1) \
                for i in range(n_classes)], axis=0)
        self.likelihoods = l / (np.bincount(y).reshape(-1,1) + 2)
        assert np.all(self.likelihoods > 0)
        assert np.all(self.likelihoods < 1)

        self.rng = np.random.RandomState(seed=43)

    def generate_sentence(self, nb_class):
        return ' '.join([self.features[i] for i in np.nonzero(
            self.rng.binomial(1, p=self.likelihoods[nb_class]))[0]])

    def predict_logits(self, X):
        X = self.cv.transform(X)
        X.data[:] = 1
        return unnormalized_logposteriors_to_logits(
            X.dot(np.log(self.likelihoods).T) - \
            X.dot(np.log(1-self.likelihoods).T) + \
            (np.log(self.priors) + np.sum(np.log(1-self.likelihoods), axis=1)).reshape(1,-1))

    def predict(self, X):
        return np.argmax(self.predict_logits(X), axis=1)

##### ZADANIE DODATKOWE NIEPUNKTOWANE #####

class MultinomialNaiveBayes:

    def __init__(self, X, y):
        n_classes = int(max(y) + 1)
        assert set(y) == set(np.arange(n_classes))

        self.priors = np.bincount(y) / len(y)

        if hasattr(X, "todense"):
            stack = np.concatenate
        else:
            stack = np.stack
        l = stack([
            np.array((X[y==i].sum(axis=0)) + 1) \
                for i in range(n_classes)], axis=0)
        self.likelihoods = l / l.sum(axis=1).reshape(-1,1)

        self.rng = np.random.RandomState(seed=43)

    def generate(self, nb_class, length):
        indices = self.rng.choice(
            np.arange(self.likelihoods.shape[1]),
            size=length,
            p=self.likelihoods[nb_class])
        result = np.zeros(self.likelihoods.shape[1])
        result[indices] = 1
        return result

    def predict_logits(self, X):
        return unnormalized_logposteriors_to_logits(
            X.dot(
                np.log(self.likelihoods).T) + \
                    np.log(self.priors).reshape(1,-1))

    def predict(self, X):
        return np.argmax(self.predict_logits(X), axis=1)

class BernoulliNaiveBayes:

    def __init__(self, X, y):
        n_classes = int(max(y) + 1)
        assert set(y) == set(np.arange(n_classes))

        self.priors = np.bincount(y) / len(y)

        X = X.copy()
        if hasattr(X, "todense"):
            X.data[:] = 1
            stack = np.concatenate
        else:
            X[X>0] = 1
            stack = np.stack
        l = stack([
            np.array((X[y==i].sum(axis=0)) + 1) \
                for i in range(n_classes)], axis=0)
        self.likelihoods = l / (np.bincount(y).reshape(-1,1) + 2)
        assert np.all(self.likelihoods > 0)
        assert np.all(self.likelihoods < 1)

        self.rng = np.random.RandomState(seed=43)

    def generate(self, nb_class):
        return self.rng.binomial(1, p=self.likelihoods[nb_class])

    def predict_logits(self, X):
        X = X.copy()
        if hasattr(X, "todense"):
            X.data[:] = 1
        else:
            X[X>0] = 1
        return unnormalized_logposteriors_to_logits(
            X.dot(np.log(self.likelihoods).T) - \
            X.dot(np.log(1-self.likelihoods).T) + \
            (np.log(self.priors) + np.sum(np.log(1-self.likelihoods), axis=1)).reshape(1,-1))

    def predict(self, X):
        return np.argmax(self.predict_logits(X), axis=1)
