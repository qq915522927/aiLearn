import numpy as np
from collections import Counter
from .metrics import accuracy_score


class KNNClassfier:

    def __init__(self, k):
        self.k = k
        # capital case means Matrix
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        assert X_train.shape[0] == y_train.shape[0], f'The count of train and target must the same {X_train.shape[0]} != {y_train.shape[0]}'
        self._X_train = X_train
        self._y_train = y_train

    def predict(self, X_predict):
        assert X_predict.shape[1] == self._X_train.shape[1], 'the counter of the charactar must be the same'
        predicts = []
        for row in X_predict:
            predicts.append(self._predict(row))

        return np.array(predicts)


    def _predict(self, characters):
        distinces = np.sqrt(np.sum((self._X_train - characters)**2, axis=1))
        nearst = np.argsort(distinces)[:self.k]
        voter = Counter(self._y_train[nearst])
        return voter.most_common(1)[0][0]

    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)


if __name__ == '__main__':
    raw_data_X = [[3.393533211, 2.331273381],
                  [3.110073483, 1.781539638],
                  [1.343808831, 3.368360954],
                  [3.582294042, 4.679179110],
                  [2.280362439, 2.866990263],
                  [7.423436942, 4.696522875],
                  [5.745051997, 3.533989803],
                  [9.172168622, 2.511101045],
                  [7.792783481, 3.424088941],
                  [7.939820817, 0.791637231]
                  ]
    raw_data_y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    raw_data_X = np.array(raw_data_X)
    raw_data_y = np.array(raw_data_y)
    x_predict = np.array([ 8.09360732,  3.36573151])
    X_predict = x_predict.reshape(1, -1)
    KNN_classifier = KNNClassfier(6)
    KNN_classifier.fit(raw_data_X, raw_data_y )
    print(KNN_classifier.predict(X_predict))