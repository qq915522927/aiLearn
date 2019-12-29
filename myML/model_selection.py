import numpy as np

def train_test_split(X, y, test_ratio=0.2, seed=None):
    assert X.shape[0] == y.shape[0], "the size of X, y must be the same"
    assert 0.0< test_ratio < 1, "Test ratio must be valid"
    if seed:
        np.random.seed(seed)

    # generate random index permulation
    shuffled_indexes = np.random.permutation(len(X))

    test_size = int(len(X) * test_ratio)

    test_indexs = shuffled_indexes[:test_size]
    train_indexs = shuffled_indexes[test_size:]

    X_train = X[train_indexs]
    y_train = y[train_indexs]

    X_test = X[test_indexs]
    y_test = y[test_indexs]

    return X_train, y_train, X_test, y_test



