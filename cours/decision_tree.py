import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# Example data
x = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.sin(x).ravel() + np.random.normal(0, 0.1, x.shape[0])

# Fit decision tree
tree = DecisionTreeRegressor(max_depth=2)
tree.fit(x, y)
y_pred = tree.predict(x)

# Plot
plt.scatter(x, y, color='lightgray', label='Data')
plt.plot(x, y_pred, color='red', label='Decision Tree Prediction')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
