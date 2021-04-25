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
    n_users, n_movies = int(max(data[:, 0])), int(max(data[:, 1]))
    M = np.zeros((n_users, n_movies), dtype=data.dtype)
    for row in data:
        user, movie, rating = row
        M[int(user) - 1, int(movie) - 1] = rating
    # M[row_pos, col_pos] = data[:, 2]

    # Initialize objective results and user/object matrices
    objective = np.zeros((iterations, 1))
    v = np.random.normal(0, np.sqrt(1/lam), (n_movies, d))
    u = np.zeros((n_users, d))
    iter_u = []
    iter_v = []

    # Learn objective
    for iter in range(iterations):
        for user in range(n_users):
            # index = np.where(users == user)[0][0]
            # V matrix for objects that have been rated by user "user"
            objs = np.nonzero(M[user, :])
            this = v[objs]
            left = np.linalg.inv((lam * variance * np.eye(d)) + xTx(this))
            right = (this * M[user, objs][:].T).sum(axis=0)
            u[user, :] = left.dot(right)
        iter_u.append(u.copy())

        for obj in range(n_movies):
            # index = np.where(movies == obj)[0][0]
            # U matrix for users that have rated object "obj"
            usrs = np.nonzero(M[:, obj])
            this = u[usrs]
            left = np.linalg.inv((lam * variance * np.eye(d)) + xTx(this))
            right = (this * M[usrs, obj][:].T).sum(axis=0)
            v[obj, :] = left.dot(right)
        iter_v.append(v.copy())

        sum_u, sum_v = map(lambda m: ((lam / 2) * np.linalg.norm(m, axis=1) ** 2).sum(), [u, v])
        c = 0
        for usr, obj, rtg in data:
            c += (rtg - np.dot(u[int(usr) - 1, :], v[int(obj) - 1, :])) ** 2
        c /= 2 * variance
        objective[iter] = -c - sum_u - sum_v

    return objective, iter_u, iter_v


def save_pmf_output(loss, u_matrices, v_matrices, iterations):
    # A comma-separated file containing the PMF objective function along each row
    np.savetxt('objective.csv', loss, delimiter=',', fmt='%1.10f')
    # A CSV for each iteration specified with the locations corresponding to each row ("user") of M, "U-{i_n}"
    # and the locations corresponding to the objects of the missing matrix M, "V-{i_n}"
    for i_n in iterations:
        np.savetxt('U-{}.csv'.format(i_n), u_matrices[i_n - 1], delimiter=',', fmt='%1.10f')
        np.savetxt('V-{}.csv'.format(i_n), v_matrices[i_n - 1], delimiter=',', fmt='%1.10f')


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

