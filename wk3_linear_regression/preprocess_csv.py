"""
Not required for submission, this translates a given CSV with independent and dependent variables into the required
data format files, plus a configurable test/train split.
"""

import os
import numpy as np

test_train_split = 0.3
y_var_loc = 0

full = np.genfromtxt(r'data\wine.csv', skip_header=1, delimiter=',')

dim = len(full)
n_train = int(dim * test_train_split)
X_test, X_train = full[:n_train, :], full[n_train + 1:, :]
y, X, Xt = X_train[:, 0], X_train[:, 1:], X_test[:, 1:]

np.savetxt(r'data\X_test.csv', Xt, fmt='%d', delimiter=',')
np.savetxt(r'data\X_train.csv', X, fmt='%d', delimiter=',')
np.savetxt(r'data\y_train.csv', y, fmt='%d', delimiter=',')
