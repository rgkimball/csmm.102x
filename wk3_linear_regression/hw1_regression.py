"""
Submission for Project 1 of Columbia University's ML EdX course (Linear Regression).

    author: @rgkimball
    date: 2/16/2021
"""

# Std Lib
import sys
# PyPI
import numpy as np

lambda_input = int(sys.argv[1])
sigma2_input = float(sys.argv[2])
X_train = np.genfromtxt(sys.argv[3], delimiter=',')
y_train = np.genfromtxt(sys.argv[4])
X_test = np.genfromtxt(sys.argv[5], delimiter=',')


def part1(x, y, lmda):
    """
    Implements closed-form solution to the ridge regression.

    :param x: dxn matrix of input covariates
    :param y: nx1 matrix of target dependent variable
    :param lmda: float, regularization parameter
    :return: dx1 matrix of ridge regression coefficients
    """
    reg = lmda * np.eye(x.shape[1])
    wts = np.dot(np.linalg.inv(reg + x.T.dot(x)), x.T.dot(y))
    return wts


wRR = part1(X_train, y_train, lambda_input)  # Assuming wRR is returned from the function
np.savetxt('wRR_{}.csv'.format(lambda_input), wRR, delimiter='\n')


# Solution for Part 2
def part2():
    # Input : Arguments to the function
    # Return : active, Final list of values to write in the file
    return 0


active = part2()  # Assuming active is returned from the function
np.savetxt('active_{}_{}.csv'.format(lambda_input, int(sigma2_input)), active, delimiter=',')
