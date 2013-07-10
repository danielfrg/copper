import numpy as np


def rmsle(y_test, y_pred):
    ans = np.log1p(y_pred) - np.log1p(y_test)
    ans = np.power(ans, 2)
    return np.sqrt(ans.mean())
