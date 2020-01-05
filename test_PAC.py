import numpy as np
import matplotlib.pyplot as plt


def demean(X):
    return X - np.mean(X, axis=0)

def f(w, X):
    return np.sum(X.dot(w) ** 2) / len(X)


def df_math(X, w):
    return X.T.dot(X.dot(w)) * 2 / len(X)


def direction(w):
    return w / np.linalg.norm(w)





def gradient_ascent(df, X, intial_w, eta, n_iters=1e4, epsilon=1e-8):
    w = direction(intial_w)
    n = 0
    while n < n_iters:
        w = direction(w)
        last_w = w
        gradient = df(X, w)
        w = w + eta * gradient
        if (abs(f(w, X) - f(last_w, X)) < epsilon):
            break
        n += 1
    return w




def test_simple_pca():
    X = np.empty((100, 2))
    X[:, 0] = np.random.uniform(0., 100, size=100)
    X[:, 1] = X[:, 0] * 0.75 + 3 + np.random.normal(0, 10, size=len(X))
    X = demean(X)
    plt.scatter(X[:, 0], X[:, 1]);
    plt.show()
    np.mean(X)
    np.mean(X, axis=0)
    initial_w = np.random.random(X.shape[1])
    eta = 1e-3
    w = gradient_ascent(df_math, X, initial_w, eta)
    plt.scatter(X[:, 0], X[:, 1])
    plt.plot([0, w[0] * 20], [0, w[1] * 20])
    plt.show()


    X2 = np.empty((100, 2))
    X2[:, 0] = np.random.uniform(0, 100, size=100)
    X2[:, 1] = X2[:, 0] * 0.8 + 3
    X2 = demean(X2)
    w = gradient_ascent(df_math, X2, initial_w, eta)
    plt.scatter(X2[:, 0], X2[:, 1])
    plt.plot([0, w[0] * 20], [0, w[1] * 20])
    plt.show()
if __name__ == '__main__':
    test_simple_pca()
