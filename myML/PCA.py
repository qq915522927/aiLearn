import numpy as np

class PCA:

    def __init__(self, n_components):
        self.n_components = n_components
        self.componens_ = None

    def fit(self, X, eta=0.01, n_iters=1e4):
        X_pca = demean(X)
        self.componens_ = get_first_n_components(self.n_components, X_pca, eta=eta, n_iters=n_iters)
        return self

    def transform(self, X):
        assert X.shape[1] == self.componens_.shape[1]
        return X.dot(self.componens_.T)

    def reverse_transform(self, X):
        assert X.shape[1] == self.componens_.shape[0]
        return X.dot(self.componens_)

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
        last_w = w
        gradient = df(X, w)
        w = w + eta * gradient
        w = direction(w)
        if (abs(f(w, X) - f(last_w, X)) < epsilon):
            break
        n += 1
    return w

def get_first_n_components(n, X, eta, n_iters=1e4, epsilon=1e-8):
    assert n <= X.shape[1], "n of components must less or equal X's demession"
    cur_n = 0
    initial_w = np.random.random(X.shape[1])
    X_cur = X.copy()
    res = []
    while cur_n < n:
        w = gradient_ascent(df_math, X_cur, initial_w, eta)
        res.append(w)
        # for i in range(len(X_cur)):
        #     X_cur[i] = X_cur[i] - X_cur[i].dot(w) * w
        X_cur = X_cur - X_cur.dot(w).reshape((-1, 1))*w
        cur_n += 1
    return np.array(res)
