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
    :return: writes a file 'centroids-{iterations}.csv' to the working directory.
    """
    # Randomly initialize the centroids within the data
    centroids = data[np.random.choice(data.shape[0], size=clusters, replace=False)]

    for i in range(iterations):
        # Calculate cluster assignments
        assignments = np.array([np.linalg.norm(x_i - centroids, axis=1, ord=2).argmin() for x_i in data])
        # Update centroids to mean of each cluster
        for a in np.unique(assignments):
            cluster_data = data[np.where(assignments == a)]
            centroids[a, :] = np.mean(cluster_data, axis=0)
        save(centroids, 'centroids-{}'.format(i + 1))


def em_gmm(data, clusters, iterations):

    save('pi-{}'.format(i + 1), pi)
    save('mu-{}'.format(i + 1), mu)  # this must be done at every iteration

    for j in range(k):  # k is the number of clusters
        # this must be done 5 times (or the number of clusters) for each iteration
        save('Sigma-{}-{}'.format(j + 1, i + 1), sigma[j])


if __name__ == '__main__':
    try:
        X = np.genfromtxt(sys.argv[1], delimiter=",")
    except IndexError:
        raise ValueError('Invalid or missing argument specifying the source data. Usage: hw3_clustering.py X.csv')

    algos = [
        k_means,
        # em_gmm,
    ]

    for algo in algos:
        algo(X, clusters=5, iterations=10)
