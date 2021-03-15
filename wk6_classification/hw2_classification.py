"""
Submission for Project 1 of Columbia University's ML EdX course (Linear Regression).

    author: @rgkimball
    date: 3/10/2021
"""

# Std Lib
from __future__ import division
import numpy as np
import sys

X_train = np.genfromtxt(sys.argv[1], delimiter=",")
y_train = np.genfromtxt(sys.argv[2])
X_test = np.genfromtxt(sys.argv[3], delimiter=",")


def plugin_classifier(train_data, train_labels, test_data):
    # this function returns the required output
    return 0


final_outputs = plugin_classifier(X_train, y_train, X_test)

np.savetxt("probs_test.csv", final_outputs, delimiter=",")  # write output to file
