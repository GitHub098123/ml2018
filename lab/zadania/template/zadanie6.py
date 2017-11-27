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
        ...

    @property
    def priors(self):
        ...

    @property
    def likelihoods(self):
        ...

    @property
    def features(self):
        ...

    def generate_sentence(self, nb_class, length):
        ...

    def predict_logits(self, X):
        return unnormalized_logposteriors_to_logits(...)

    def predict(self, X):
        return np.argmax(self.predict_logits(X), axis=1)

class TextBernoulliNaiveBayes:

    def __init__(self, X, y):
        ...

    @property
    def priors(self):
        ...

    @property
    def likelihoods(self):
        ...

    @property
    def features(self):
        ...

    def generate_sentence(self, nb_class):
        ...

    def predict_logits(self, X):
        return unnormalized_logposteriors_to_logits(...)

    def predict(self, X):
        return np.argmax(self.predict_logits(X), axis=1)

##### ZADANIE DODATKOWE NIEPUNKTOWANE #####

class MultinomialNaiveBayes:

    def __init__(self, X, y):
        ...

    @property
    def priors(self):
        ...

    @property
    def likelihoods(self):
        ...

    def generate(self, nb_class, length):
        ...

    def predict_logits(self, X):
        return unnormalized_logposteriors_to_logits(...)

    def predict(self, X):
        return np.argmax(self.predict_logits(X), axis=1)

class BernoulliNaiveBayes:

    def __init__(self, X, y):
        ...

    @property
    def priors(self):
        ...

    @property
    def likelihoods(self):
        ...

    def generate(self, nb_class):
        ...

    def predict_logits(self, X):
        return unnormalized_logposteriors_to_logits(...)

    def predict(self, X):
        return np.argmax(self.predict_logits(X), axis=1)
