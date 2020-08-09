from numba import jit
import numpy as np


# @jit(nopython=True)
def make_one_hot(y, n_class, n_samples):
    # print(y, n_class, n_samples)
    result = np.zeros((n_samples, n_class), np.int64)
    for i in range(n_samples):
        result[i, y[i]] = 1
    return result


if __name__ == "__main__":
    y = [11595, 11595, 11595, 11595, 11595, 11595, 11595, 11595, 11595, 11595, 11595, 11595,
         11595, 11595, 11595, 11595, 11595, 11595, 11595, 11595, 11595, 11595, 11595, 11595,
         11595, 11595, 11595, 11595, 11595, 11595, 11595, 11595, 11595, 11595]

    print(make_one_hot(y=y, n_class=1000, n_samples=34))
