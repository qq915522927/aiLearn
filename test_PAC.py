import time
import numpy as np
from myML.PCA import PCA
import matplotlib.pyplot as plt
from sklearn import datasets


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

def get_first_n_components(n, X, initial_w, eta, n_iters=1e4, epsilon=1e-8):
    assert n <= X.shape[1], "n of components must less or equal X's demession"
    cur_n = 0
    initial_w = np.random.random(X.shape[1])
    X_cur = X.copy()
    res = []
    while cur_n < n:
        w = gradient_ascent(df_math, X_cur, initial_w, eta)
        res.append(w)
        for i in range(len(X_cur)):
            X_cur[i] = X_cur[i] - X_cur[i].dot(w) * w
        cur_n += 1
    return np.array(res)



def test_simple_pca():
    X = np.empty((100, 2))
    X[:, 0] = np.random.uniform(0., 100, size=100)
    X[:, 1] = X[:, 0] * 0.75 + 3 + np.random.normal(0, 10, size=len(X))
    X = demean(X)
    plt.scatter(X[:, 0], X[:, 1]);
    plt.show()
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

def test_get_first_n_components():
    X = np.empty((100, 2))
    X[:, 0] = np.random.uniform(0., 100, size=100)
    X[:, 1] = X[:, 0] * 0.75 + 3 + np.random.normal(0, 10, size=len(X))
    X = demean(X)
    initial_w = np.random.random(X.shape[1])
    eta = 1e-3
    components = get_first_n_components(2, X, initial_w, eta)
    print(components)
    print(components[0].dot(components[1]))
    plt.scatter(X[:,0], X[:, 1])
    plt.plot([0, components[0][0] *20], [0, components[0][1]*20], color='r')
    plt.plot([0, components[1][0] *20], [0, components[1][1]*20], color='g')
    plt.show()

def test_my_pca():
    X = np.empty((100, 2))
    X[:, 0] = np.random.uniform(0., 100, size=100)
    X[:, 1] = X[:, 0] * 0.75 + 3 + np.random.normal(0, 10, size=len(X))
    X = demean(X)
    initial_w = np.random.random(X.shape[1])
    eta = 1e-3
    pca = PCA(1)
    pca.fit(X, eta)

    plt.scatter(X[:, 0], X[:,1])

    X_trans = pca.transform(X)
    plt.scatter(X_trans[:,0], np.ones(len(X_trans)))

    X_reverse = pca.reverse_transform(X_trans)
    plt.scatter(X_reverse[:, 0], X_reverse[:,1])
    plt.show()

def test_sklearn_pca():
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier()
    t1 = time.time()
    clf.fit(X_train, y_train)
    score1 = clf.score(X_test, y_test)
    t2 = time.time()
    print(f"Directly get score: {score1} , elapsed: {t2-t1}")

    pca = PCA(n_components=2)
    pca.fit(X_train)
    X_train_redu = pca.transform(X_train)
    X_test_redu = pca.transform(X_test)

    for i in range(10):
        plt.scatter(X_train_redu[y_train==i, 0], X_train_redu[y_train==i,1])
    plt.show()


    clf = KNeighborsClassifier()
    t1 = time.time()
    clf.fit(X_train_redu, y_train)
    score2 = clf.score(X_test_redu, y_test)
    t2 = time.time()
    print(f"Use PCA component=2 get score: {score2} , elapsed: {t2 - t1}")

    pca = PCA(n_components=X_train.shape[1])
    pca.fit(X_train)
    plt.plot(
        [i for i in range(len(pca.explained_variance_ratio_))],
        [np.sum(pca.explained_variance_ratio_[:i+1]) for i in range(X_train.shape[1])]
    )
    plt.show()

    pca = PCA(0.95)
    pca.fit(X_train)
    print(f"Component num for 0.95 PCA: {pca.n_components_}")
    X_train_redu = pca.transform(X_train)
    X_test_redu = pca.transform(X_test)
    clf = KNeighborsClassifier()
    t1 = time.time()
    clf.fit(X_train_redu, y_train)
    score2 = clf.score(X_test_redu, y_test)
    t2 = time.time()
    print(f"Use PCA component={pca.n_components_} get score: {score2} , elapsed: {t2 - t1}")

if __name__ == '__main__':
    test_sklearn_pca()
