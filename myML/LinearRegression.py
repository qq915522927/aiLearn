import traceback
import numpy as np
import matplotlib.pyplot as plt
from .metrics import r2_score

class LinearRegression:

    def __init__(self):
        # xi shu xiang liang
        self.coef_ = None
        # jie ju
        self.intercept_ = None
        self._theta = None

    def fit_math(self, X_train, y_train):
        # use math to cal the theta and intercept
        assert X_train.shape[0] == y_train.shape[0], "the size must be equal"

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
        self.coef_ = self._theta[1:]
        self.intercept_ = self._theta[0]

    def fit_gd(self, X_train, y_train, eta=0.01, n_iters=1e5):
        X_b = np.hstack([np.full((len(X_train), 1), 1), X_train])
        # initial_theta = np.random.normal(size=X_train.shape[1] + 1)
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_desent(X_b, y_train, initial_theta, eta, n_iters)
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self

    def fit_rand_gd(self, X_train, y_train, eta=0.01, n_iters=5):
        X_b = np.hstack([np.full((len(X_train), 1), 1), X_train])
        # initial_theta = np.random.normal(size=X_train.shape[1] + 1)
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = random_gradient_desent(X_b, y_train, initial_theta, eta, n_iters)
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self

    def predict(self, X_predict):
        assert X_predict.shape[1] == len(self.coef_), 'must be the same demesion'
        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self._theta.T)

    def score(self, x_test, y_test):
        assert x_test.shape[0] == len(y_test)
        y_predict = self.predict(x_test)
        return r2_score(y_test, y_predict)

def J(X_b, y, theta):
    try:
        return np.sum((y - X_b.dot(theta)) ** 2) / len(X_b)
    except Exception as e:
        traceback.print_exc()
        print(e)
        return float('inf')
def dJ_loop(X_b, y, theta):
    res = np.empty(len(theta))
    for i in range(len(theta)):
        res[i] = (X_b.dot(theta) - y).dot(X_b[:, i]) * 2/ len(X_b)
    return res

def dJ(X_b, y, theta):
    res = X_b.T.dot(X_b.dot(theta) - y) * 2 / len(X_b)
    return res

def dJ_debug(X_b, y, theta, epsilon=0.01):
    res = np.empty(len(theta))
    for i in range(len(theta)):
        theta1 = theta.copy()
        theta1[i] -= epsilon
        theta2 = theta.copy()
        theta2[i] += epsilon
        res[i] = (J(X_b, y, theta2) - J(X_b, y, theta1)) / (2 * epsilon)
    return res


def gradient_desent(X_b, y, intial_theta, eta, n_iters, epsilon=1e-8):
    n = 0
    theta = intial_theta
    while n <= n_iters:
        last_theta = theta
        gradient = dJ(X_b, y, theta)
        theta = theta - gradient * eta
        if(abs(J(X_b, y, theta) - J(X_b, y, last_theta)) < epsilon):
            break
        n += 1
    # for i in range(0, len(history), 100):
    #     plt.plot(range(len(history[i])), history[i])
    # plt.show()
    return theta

def random_gradient_desent(X_b, y, intial_theta, eta, n_iters=5, epsilon=1e-8):
    """
    Randomly gradient desent
    """
    n = 0
    theta = intial_theta
    window_size = int(len(X_b) / 100);
    while n <= n_iters:
        index = 0
        ran_row = np.random.permutation(len(X_b))
        while(index < len(y)):
            last_theta = theta
            window_indexs = ran_row[index: index + window_size]
            gradient = dJ(X_b[window_indexs], y[window_indexs], theta)
            theta = theta - gradient * eta
            index += window_size
            if(abs(J(X_b, y, theta) - J(X_b, y, last_theta)) < epsilon):
                return theta
        n += 1
    # for i in range(0, len(history), 100):
    #     plt.plot(range(len(history[i])), history[i])
    # plt.show()
    return theta
