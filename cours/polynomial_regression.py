import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# np.random.seed(42)
# X = np.linspace(0, 10, 80).reshape(-1, 1)
# Y = 2 + 1.5 * X.flatten() + 0.8 * X.flatten()**2 + np.random.randn(80) * 5

X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
Y = np.array([2, 3, 5, 8, 12, 18])

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, Y)

X_plot = np.linspace(0, 7, 100).reshape(-1, 1)
X_plot_poly = poly.transform(X_plot)
Y_pred = model.predict(X_plot_poly)


plt.figure(figsize=(10, 6))
plt.scatter(X, Y, color='blue', label='Actual Data')
plt.plot(X_plot, Y_pred, color='red', label='Polynomial Regression (degree=2)')
plt.title('Polynomial Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)

# Evaluation
# r2 = r2_score(Y, model.predict(X_poly))
# plt.text(0.5, 16, f'R² Score: {r2:.3f}', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

plt.show()
