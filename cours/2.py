import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression


df = pd.read_csv('dataset.csv',delimiter=";")
x = df.iloc[:,0].values
y = df.iloc[:,-1].values
x = x.reshape(len(x),1)
y = y.reshape(len(y),1)
#coeficient de determination
model = LinearRegression()
model.fit(x,y)
print("Coefficient de determination: ", model.score(x,y))
plt.scatter(x,y,color="b")
plt.show()