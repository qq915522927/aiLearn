

def accuracy_score(y_true, y_predict):
    assert y_true.shape[0] == y_predict.shape[0], "the szie must be the same"
    return sum(y_true==y_predict) / len(y_true)