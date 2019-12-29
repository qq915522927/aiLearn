import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt
from myML.model_selection import train_test_split
from myML.KNN import KNNClassfier
from myML.preprocessing import StandardScaler

def test_iris():
    iris = datasets.load_iris()
    test_dataset(iris, 'Iris')

def test_dataset(dateset, name):
    print(f"------test for {name} ---------")
    X = dateset.data
    y = dateset.target
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_ratio=0.2)
    classifier = KNNClassfier(6)
    classifier.fit(X_train, y_train)
    print(classifier.score(X_test, y_test))

    signal_predic_y = classifier.predict(X_test[0].reshape(1, -1))
    print(f"Test for y_test: {y_test[0]} is {signal_predic_y[0]}")


def test_digit():
    digits = datasets.load_digits()
    test_dataset(digits, 'Digits')

def test_hyper_parameters():
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=55)

    best_score = 0.0
    best_k = -1
    best_method = ""
    for p in range(1, 6):
        for method in ['uniform', 'distance']:
            for k in range(1, 10):
                knn_clf = KNeighborsClassifier(n_neighbors=k, weights=method, p=p)
                knn_clf.fit(X_train, y_train)
                score = knn_clf.score(X_test, y_test)
                if score > best_score:
                    best_score = score
                    best_k = k
                    best_method = method
    print(f"Best score: {best_score}")
    print(f"Best K: {best_k}")
    print(f"Best method: {best_method}")

def test_grid_search():
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import GridSearchCV
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=55)
    param_grid = [
        {
            'weights': ['uniform'],
            'n_neighbors': [i for i in range(1, 11)]
        },
        {
            'weights': ['distance'],
            'n_neighbors': [i for i in range(1, 11)],
            'p': [i for i in range(1, 6)]
        }
    ]
    knn_clf = KNeighborsClassifier()
    grid_search = GridSearchCV(knn_clf, param_grid, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)


def test_my_standardScaler():
    scaler = StandardScaler()
    iris = datasets.load_iris()
    X = iris.data
    scaler.fit(X)
    std_X = scaler.transform(X)
    plt.scatter(std_X[:, 0], std_X[:,1], np.full(std_X.shape[0], 100))
    plt.show()




if __name__ == '__main__':
    test_my_standardScaler()
