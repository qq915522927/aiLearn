import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from myML.SimpleLinearRegression import SimpeLinearRegression


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



if __name__ == '__main__':
    test_simple_linear_regression1()
