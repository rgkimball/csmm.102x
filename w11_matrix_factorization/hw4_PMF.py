"""
Submission for Project 4 of Columbia University's ML EdX course (Probabilistic Matrix Factorization).

    author: @rgkimball
    date: 4/25/2021
"""

import os
import sys
import numpy as np


def generate_data(destination, n_users=100, rating_range=range(1, 5), n_movies=100, n_ratings_range=range(3, 20)):
    movies = np.random.randint(1, n_movies + 1, n_movies)
    users = range(1, n_users + 1)
    ratings = []
    for user in users:
        for _ in range(np.random.choice(n_ratings_range, 1)[0]):
            ratings.append([user, np.random.choice(movies, 1)[0], np.random.choice(rating_range, 1)[0]])
    np.savetxt(destination, np.array(ratings), delimiter=',', fmt='%i')


def decr(x):
    """Decrement an integer by 1"""
    return int(x) - 1


def xTx(mat):
    return mat.T.dot(mat)


def pmf(data, lam=2, variance=0.1, d=5, iterations=50):
    """
    Runs probabilistic matrix factorization algorithm on a dataset of sparse user-object ratings.

    :param data: n by 3 matrix, with columns: user id, object id, and rating (continuous or discrete value)
    :param lam: lambda parameter
    :param variance: variance parameter
    :param d: dimensions
    :param iterations: number of iterations to refine objective function over
    :return: tuple, the loss, the user matrix and the object matrix over all iterations.
    """
    # Pivot data into i by j matrix of users (rows) and movies (columns) with ratings (values)
    users, row_pos = np.unique(data[:, 0], return_inverse=True)
    movies, col_pos = np.unique(data[:, 1], return_inverse=True)
    M = np.zeros((np.amax(data[:, 0]), np.amax(data[:, 1])), dtype=data.dtype)
    M[row_pos, col_pos] = data[:, 2]

    # Initialize objective results and user/object matrices
    objective = np.zeros((iterations, 1))
    v = np.random.normal(0, np.sqrt(1/lam), (len(movies), d))
    u = np.zeros((len(users), d))
    iter_u = []
    iter_v = []

    # Learn objective
    for iter in range(iterations):
        for user in map(int, users):
            index = np.where(users == user)[0][0]
            # V matrix for objects that have been rated by user i
            objs = np.nonzero(M[index, :])
            this = v[objs]
            left = np.linalg.inv((lam * variance * np.eye(d)) + xTx(this))
            right = (this * M[index, objs][:].T).sum(axis=0)
            u[index, :] = left.dot(right)
        iter_u.append(u.copy())

        for obj in map(int, movies):
            index = np.where(movies == obj)[0][0]
            usrs = np.nonzero(M[:, index])
            this = u[usrs]
            left = np.linalg.inv((lam * variance * np.eye(d)) + xTx(this))
            right = (this * M[usrs, index][:].T).sum(axis=0)
            v[index, :] = left.dot(right)
        iter_v.append(v.copy())

        t2, t3 = map(lambda m: ((lam / 2) * np.linalg.norm(m, axis=1) ** 2).sum(), [u, v])
        t1 = 0
        for usr, obj, rtg in data:
            i = np.where(users == usr)[0][0]
            j = np.where(movies == obj)[0][0]
            t1 += (rtg - np.dot(u[i - 1, :], v[j - 1, :])) ** 2
        t1 /= 2 * variance
        objective[iter] = -t1 - t2 - t3

    return objective, iter_u, iter_v


def save_pmf_output(loss, u_matrices, v_matrices, iterations):
    # A comma-separated file containing the PMF objective function along each row
    np.savetxt('objective.csv', loss, delimiter=',', fmt='%1.2f')
    # A CSV for each iteration specified with the locations corresponding to each row ("user") of M, "U-{i_n}"
    # and the locations corresponding to the objects of the missing matrix M, "V-{i_n}"
    for i_n in iterations:
        np.savetxt('U-{}.csv'.format(i_n), u_matrices[i_n - 1], delimiter=',', fmt='%1.5f')
        np.savetxt('V-{}.csv'.format(i_n), v_matrices[i_n - 1], delimiter=',', fmt='%1.5f')


if __name__ == '__main__':
    data_path = sys.argv[1]
    if not os.path.isfile(data_path):
        print('Generating random data for testing, file does not exist at path ' + data_path)
        generate_data(data_path)
    train_data = np.genfromtxt(data_path, delimiter=',')

    # Assuming the PMF function returns Loss L, U_matrices and V_matrices (refer to lecture)
    output = pmf(train_data)
    save_pmf_output(*output, iterations=[10, 25, 50])
    print('Done')

