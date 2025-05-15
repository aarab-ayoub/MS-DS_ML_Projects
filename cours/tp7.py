#regression linear avec gradient descent et sgdregressor()_personnels 

import pandas as pd
from sklearn.linear_model import SGDRegressor
import matplotlib.pyplot as plt

df = pd.read_csv('dataset.csv',delimiter=";")
# print(df.head())
x = df.iloc[:,0].values
y = df.iloc[:,-1].values
x = x.reshape(len(x), 1)
y = y.reshape(len(y), 1)
plt.scatter(x, y, color="b")

model = SGDRegressor(max_iter=1000, tol=1e-3)
model.fit(x, y)

r = model.score(x, y)
print(r*100 , "%")
y_pred =model.predict(x)

plt.scatter(x, y, color="r")
plt.plot(x, y_pred, color="g")
plt.title("Regression line")
plt.show()
