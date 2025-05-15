import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

# Generate some data
x, y = make_regression(n_samples=100, n_features=1, noise=5)
y = y.reshape(x.shape[0], 1)

def F(x, theta):
    return x.dot(theta)

def FonnctionCout(x, y, theta):
    m = len(x)
    return (1 / (2 * m)) * np.sum((F(x, theta) - y) ** 2)

def gradient(x, y, theta):
    m = len(x)
    return (1 / m) * x.T.dot(F(x, theta) - y)

def gradient_descent(x, y, theta, alpha=0.001, iterations=2000):
    for i in range(iterations):
        theta = theta - alpha * gradient(x, y, theta)
    return theta


def gradient_descent_(x ,y, theta, alpha=0.001, iterations=2000):
	thetaHistoire = np.zeros((iterations, 2))

# def coefficients(x, y, theta, alpha=0.001, iterations=2000):
	# A =((y - y_prediction) ** 2).sum()
theta = np.zeros((x.shape[1], 1))

theta = gradient_descent(x, y, theta)

y_prediction = F(x, theta)

plt.scatter(x[:, 0], y)
plt.plot(x[:, 0], y_prediction, color="r")
plt.show()

# plt.scatter(x[:, 0], y, color="b")
# plt.show()
