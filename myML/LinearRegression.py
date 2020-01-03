import numpy as np
from .metrics import r2_score

class LinearRegression:

    def __init__(self):
        # xi shu xiang liang
        self.coef_ = None
        # jie ju
        self.intercept_ = None
        self._theta = None

    def fit(self, X_train, y_train):
        assert X_train.shape[0] == y_train.shape[0], "the size must be equal"

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
        self.coef_ = self._theta[1:]
        self.intercept_ = self._theta[1]

    def predict(self, X_predict):
        assert X_predict.shape[1] == len(self.coef_), 'must be the same demesion'
        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self._theta.T)

    def score(self, x_test, y_test):
        assert x_test.shape[0] == len(y_test)
        y_predict = self.predict(x_test)
        return r2_score(y_test, y_predict)

