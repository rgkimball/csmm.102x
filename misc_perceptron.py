"""
Week 6, Implementation of a Quiz Question

Incomplete gradient descent since we're only training on the first 10 examples.
"""

import numpy as np
from matplotlib import pyplot as plt

data = np.array([
    [10,  0, 8,  3, 4, 0.5, 4, 2],
    [10,  0, 4,  3, 8, 0.5, 3, 5],
    [ 1, -1, 1, -1, 1,  -1, 1, 1],  # class labels
])

fig, ax = plt.subplots(figsize=(11, 8))
classes = np.unique(data[2])

for cls in classes:
    this = data.T[data[2] == cls].T
    ax.plot(this[0], this[1], marker='x' if cls > 0 else 'o', linestyle='', ms=12)


def sign(lb):
    return 1 if lb > 0 else -1


def update_weights(weights, bias, label, features):
    g = label * (weights.T.dot(features) + bias)
    if g <= 0:
        weights = weights + (label * features)
        bias += label
    return weights, bias


w = np.array([1, 1])
b = 0
xlin = np.linspace(0, 10, 1000)
for i in range(len(data.T)):
    y = data.T[i][2]
    x = data.T[i][:2]
    w, b = update_weights(w, b, y, x)
    print({
        'iteration': i,
        'label': y,
        'predicted class': sign(w.T.dot(x) + b),
        'correct?': sign(w.T.dot(x) + b) == y,
        'values': x,
        'weights': w,
        'bias': b,
    })
    ax.plot(xlin, (-w[0] / w[1]) * xlin + (-b / w[1]), c='g', ls='--', lw=1.5)
ax.plot(xlin, (-w[0] / w[1]) * xlin + (-b / w[1]), c='k', ls='-', lw=2)
plt.show()
