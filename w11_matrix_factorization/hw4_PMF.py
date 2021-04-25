"""
Submission for Project 4 of Columbia University's ML EdX course (Probabilistic Matrix Factorization).

    author: @rgkimball
    date: 4/25/2021
"""

import sys
import numpy as np


def pmf(data, lam=2, sigma2=0.01, d=5, iterations=50):
    loss, u, l = None, None, None
    return loss, u, l


def save_pmf_output(loss, u_matrices, v_matrices, iterations):
    # A comma-separated file containing the PMF objective function along each row
    np.savetxt('objective.csv', loss, delimiter=',')
    # A CSV for each iteration specified with the locations corresponding to each row ("user") of M, "U-{i_n}"
    # and the locations corresponding to the objects of the missing matrix M, "V-{i_n}"
    for i_n in iterations:
        np.savetxt('U-{}.csv'.format(i_n), u_matrices[i_n - 1], delimiter=',')
        np.savetxt('V-{}.csv'.format(i_n), v_matrices[i_n - 1], delimiter=',')


if __name__ == '__main__':
    train_data = np.genfromtxt(sys.argv[1], delimiter=',')

    # Assuming the PMF function returns Loss L, U_matrices and V_matrices (refer to lecture)
    output = pmf(train_data)
    save_pmf_output(*output)
