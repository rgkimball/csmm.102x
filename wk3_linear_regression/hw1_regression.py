"""
Submission for Project 1 of Columbia University's ML EdX course (Linear Regression).

    author: @rgkimball
    date: 2/16/2021
"""

# Std Lib
import sys
# PyPI
import numpy as np


def ridge_regression(X, y, lmda):
    """
    Implements closed-form solution to the ridge regression.

    This function is not used since lin_reg() implements parts 1 & 2, as suggested in the assignment FAQs.

    :param X: dxn matrix of input covariates
    :param y: nx1 matrix of target dependent variable
    :param lmda: float, regularization parameter
    :return: dx1 matrix of ridge regression coefficients
    """
    d = X.shape[1]
    reg = lmda * np.eye(d)
    wts = np.dot(np.linalg.inv(reg + X.T.dot(X)), X.T.dot(y))
    return wts


def lin_reg(X, y, x1, lm, s2):
    """
    Generator function to iteratively improve a L2-linear regression through Bayesian active learning.

    :param X: np.array, dxn matrix of input covariates
    :param y: np.array, nx1 matrix of target dependent variable
    :param x1: np.array, dxn matrix of test sample for iterative sampling
    :param lm: numeric, regularization parameter lambda
    :param s2: float, sigma squared
    :return: (np.array(), int)
        a tuple containing the current optimal weights and the index of the most uncertain row from the test set
    """
    d = X.shape[1]
    # Initialize autocorrelation & cross-correlation matrices
    ac = np.zeros((d, d))
    cc = np.zeros(d)
    # Initialize regularization matrix
    ld = lm * np.eye(d)

    # Preserve a copy of our test data so we can find the original data locations after we remove rows.
    s_x1 = x1.copy()
    while True:
        cc += X.T.dot(y)
        ac += X.T.dot(X)

        # Calculate weights vector & covariance
        covariance = np.linalg.inv(ld + ac / s2)
        w = np.linalg.inv(ld * s2 + ac).dot(cc)

        # Estimate uncertainty in test sample covariates and locate max variance
        variance = x1.dot(covariance).dot(x1.T).diagonal()
        loc = np.argmax(variance)
        index = np.where(np.all(s_x1 == x1[loc, :], axis=1))[0]
        x1 = np.delete(x1, loc, axis=0)

        # Recalculate our predictions based on the new covariate coefficients
        y = X.dot(w)

        yield w, int(index) + 1


if __name__ == '__main__':
    print('Running ', ' '.join(sys.argv))
    # Parse CLI arguments and load data from disk
    lambda_input = int(sys.argv[1])
    sigma2_input = float(sys.argv[2])
    X_train = np.genfromtxt(sys.argv[3], delimiter=',')
    y_train = np.genfromtxt(sys.argv[4])
    X_test = np.genfromtxt(sys.argv[5], delimiter=',')

    # Run regression and active learning iteration
    fn = lin_reg(X_train, y_train, X_test, lambda_input, sigma2_input)
    weights, active = zip(*[next(fn) for _ in range(10)])
    wRR = weights[0]

    # Save output
    np.savetxt('wRR_{}.csv'.format(lambda_input), wRR, delimiter='\n')
    np.savetxt('active_{}_{}.csv'.format(lambda_input, int(sigma2_input)), [active], fmt='%d', delimiter=',')
