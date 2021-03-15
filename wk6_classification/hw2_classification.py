"""
Submission for Project 1 of Columbia University's ML EdX course (Linear Regression).

    author: @rgkimball
    date: 3/10/2021
"""

# Std Lib
import sys
import numpy as np
from numpy.linalg import inv, det


def plugin_classifier(train_data, train_labels, test_data):
    """
    Implements plug-in classifier

    :param train_data: np.array, training feature data
    :param train_labels: 1d np.array, training labels
    :param test_data: np.array of same dimensions as train_data, test data
    :return:
    """

    # Initialize local vars & empty result matrices
    n_obs = train_labels.shape[0]
    # n_cov = train_data.shape[1]
    n_test = test_data.shape[0]
    labels, c = np.unique(train_labels, return_counts=True)
    k = len(labels)
    probabilities = np.zeros(shape=(n_test, k))

    for i in labels:
        # Select class-specific training data & labels
        x = train_data[train_labels == i]
        class_labels = train_labels[train_labels == i]
        # Simply use the fraction of class members as the MLE estimate of priors
        prior = len(class_labels) / n_obs
        mean_i = np.mean(x, axis=0)
        # De-mean the training data
        norm_x = x - mean_i
        # Calculate empirical covariance matrix
        cov = norm_x.T.dot(norm_x) / len(x)
        cov_det = det(cov) ** -0.5

        for j in range(n_test):
            x_0 = test_data[j, :]
            # Calculate how likely the new data point is to belong to each class
            p = prior * cov_det * np.exp(-0.5 * (x_0 - mean_i).T.dot(inv(cov)).dot(x_0 - mean_i))
            probabilities[j, int(i)] = p

    # Normalize for each class & return
    return np.array(list(probabilities[i, :] / sum(probabilities[i, :]) for i in range(n_test)))


if __name__ == '__main__':
    # Interpret arguments
    X_train = np.genfromtxt(sys.argv[1], delimiter=",")
    y_train = np.genfromtxt(sys.argv[2])
    X_test = np.genfromtxt(sys.argv[3], delimiter=",")

    # Calculate class probability estimates
    final_outputs = plugin_classifier(X_train, y_train, X_test)

    # Write to file
    np.savetxt("probs_test.csv", final_outputs, delimiter=',', fmt='%d')
