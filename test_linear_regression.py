import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from myML.SimpleLinearRegression import SimpeLinearRegression
from sklearn.linear_model import SGDRegressor


def test_simple_linear_regression1():
    n = 100
    x = np.array(range(1, n+1))
    y = np.array([i*0.5 + 30 for i in range(1, n+1)], dtype=float)
    for i in range(n):
        p = np.random.randint(-20, 10)
        x[i] = x[i] + p
    reg = SimpeLinearRegression()
    reg.fit(x, y)
    print(reg.a_)
    print(reg.b_)
    plt.scatter(x, y)
    plt.plot(x, reg.predict(x), color='r')
    plt.axis([0, 100, 0, 100])
    plt.show()

def test_mutiple_linear_regression():
    from sklearn.datasets import load_boston
    boston = load_boston()
    # plt.scatter(boston.data[:, 4], boston.target)
    X = boston.data[boston.target < 50]
    y = boston.target[boston.target < 50]
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=66)


    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train, y_train)
    scaler.transform(X_train)
    X_train_std = scaler.transform(X_train)

    from myML.LinearRegression import LinearRegression
    reg = LinearRegression()
    # randomly gd is much faster than the whole gd
    reg.fit_rand_gd(X_train_std, y_train, n_iters=5000, eta=0.0001)
    # reg.fit_gd(X_train_std, y_train, n_iters=500000, eta=0.0001)
    print(reg.coef_)
    print(reg.intercept_)
    X_test_std = scaler.transform(X_test)
    print(f"score:{reg.score(X_test_std, y_test)}")

    # use sgd in sklearn
    sgd_reg = SGDRegressor()
    sgd_reg.fit(X_train_std, y_train)
    print(f"Sklearn score:{sgd_reg.score(X_test_std, y_test)}")


if __name__ == '__main__':
    test_mutiple_linear_regression()
