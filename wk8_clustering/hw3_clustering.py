"""
Submission for Project 3 of Columbia University's ML EdX course (Clustering).
Implements both the K-means and EM algorithms.

    author: @rgkimball
    date: 4/18/2021
"""

import sys
import numpy as np
import scipy as sp
import pandas as pd


def cast_2d_3d(arr, n=2):
    """
    Copy data along an axis into a third dimension.

    :param arr: 2D np.array
    :param n: number of times to copy the axis into the new dimension.
    :return: 3D np.array
    """
    return np.repeat(arr[:, :, np.newaxis], n, axis=2)


def random_sample(data, n, axis=0):
    """
    Randomly select n rows from a given np.array

    :param data: np.array
    :param n: int, number of rows to select
    :param axis: optional, int - the index of the dimension along which to sample.
    :return: np.array, subset of data
    """
    return data[np.random.choice(data.shape[axis], size=n, replace=False)]


def save(data, name, ext='csv'):
    """
    Saves a numpy array to as a comma-separated file to the working directory.

    :param data: np.ndarray, 1D or 2D
    :param name: desired name of the file.
    :param ext: file extension, csv by default
    """
    return np.savetxt('{}.{}'.format(name, ext), data, delimiter=',', fmt='%.15f', newline='\n')


def k_means(data, clusters, iterations):
    """
    Implements the K-means clustering algorithm using coordinate ascent.

    :param data: np.ndarray, each row containing the x_i vector of the data to cluster
    :param clusters: int, number of desired clusters to derive from the data
    :param iterations: int, number of times to iterate through the data points to revise the classification.
    :return: writes a file 'centroids-{iteration}.csv' to the working directory for each iteration.
    """
    # Randomly initialize the centroids within the data
    centroids = random_sample(data, n=clusters)

    for i in range(iterations):
        # Calculate cluster assignments
        assignments = np.array([np.linalg.norm(x_i - centroids, axis=1, ord=2).argmin() for x_i in data])
        # Update centroids to mean of each cluster
        for a in np.unique(assignments):
            cluster_data = data[np.where(assignments == a)]
            centroids[a, :] = np.mean(cluster_data, axis=0)
        save(centroids, 'centroids-{}'.format(i + 1))


def em_gmm(data, clusters, iterations):
    """
    Implements the Gaussian Mixture Model clustering algorithm using Expectation-Maximization.

    :param data: np.ndarray, each row containing the x_i vector of the data to cluster
    :param clusters: int, number of desired clusters to derive from the data
    :param iterations: int, number of times to iterate through the data points to revise the classification.
    :return: writes several files to the working directory for each iteration:
        pi-{iteration}.csv: the cluster probabilities of the model, each row is the kth probability.
        mu-{iteration}.csv: the means of each Gaussian, each row is the kth mean.
        Sigma-{cluster}-{iteration}.csv: the covariance matrix of the Gaussian d*d for the dimensions in data.
    """

    n, d = data.shape
    phi = np.zeros((n, clusters))
    mean = random_sample(data, n=clusters)      # mu, initialized randomly from X
    prior = (1 / clusters) * np.ones(clusters)  # pi, initialized with the uniform distribution
    cov = cast_2d_3d(np.eye(d), n=clusters)     # sigma, initialized with the identity matrix

    for i in range(iterations):
        # Expectation step:
        for k in range(clusters):
            inv_cov = np.linalg.inv(cov[:, :, k])
            sqrt_det = np.linalg.det(cov[:, :, k]) ** -0.5
            for row in range(n):
                x_i = data[row, :]
                _denom = (x_i - mean[k]).T.dot(inv_cov).dot(x_i - mean[k])
                phi[row, k] = prior[k] * (2 * np.pi)**(-d/2) * sqrt_det * np.exp(_denom * -0.5)
            # Transform phi into row-wise probabilities
            phi /= phi.sum(axis=1)[:, np.newaxis]

        # Maximization step:
        nk = np.sum(phi, axis=0)
        for k in range(clusters):
            mean[k] = (phi[:, k].T.dot(data)) / nk[k]
        for k in range(clusters):
            t = np.zeros((d, d))
            for row in range(n):
                x_i = data[row, :]
                t += + phi[row, k] * np.outer(x_i - mean[k], x_i - mean[k])
            cov[:, :, k] = t / nk[k]
            save(cov[:, :, k], 'Sigma-{}-{}'.format(k + 1, i + 1))

        prior = nk/n
        save(mean, 'mu-{}'.format(i + 1))
        save(prior, 'pi-{}'.format(i + 1))


if __name__ == '__main__':
    try:
        X = np.genfromtxt(sys.argv[1], delimiter=",")
    except IndexError:
        raise ValueError('Invalid or missing argument specifying the source data. Usage: hw3_clustering.py X.csv')

    algos = [
        k_means,
        em_gmm,
    ]

    for algo in algos:
        algo(X, clusters=5, iterations=10)
