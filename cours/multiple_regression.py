import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)
X0 = np.random.rand(300) * 10 
X1 = np.random.rand(300) * 10 
Y_real = 3 * X0 + 5 * X1 + 10 + np.random.randn(300) * 2 

X = np.column_stack((X0, X1))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_real, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

# 2D Scatter plot (Real vs Predicted)
# plt.figure(figsize=(12, 6))
# plt.scatter(Y_test, Y_pred, color='royalblue', alpha=0.7)
# plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], color='red', linestyle='--')
# plt.title(f"Real vs Predicted (MSE: {mse:.2f}, R²: {r2:.2f})")
# plt.xlabel('Real Y')
# plt.ylabel('Predicted Y')
# plt.show()

# 3D Plot (X0, X1, Y)
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_test[:, 0], X_test[:, 1], Y_test, color='green', alpha=0.7, label='Real Y')
ax.scatter(X_test[:, 0], X_test[:, 1], Y_pred, color='orange', alpha=0.7, label='Predicted Y')
ax.set_title('3D View: Real vs Predicted Y')
ax.set_xlabel('X0')
ax.set_ylabel('X1')
ax.set_zlabel('Y')
ax.legend()
plt.show()
