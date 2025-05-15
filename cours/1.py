import numpy as np
import matplotlib.pyplot as plt 


x = np.linspace(0, 5, 100)

y = 2*x + 1

plt.plot(x,y,label="y = 2*x + 1",color="b")

plt.xlabel("x")
plt.ylabel("y")
plt.title("1.py")
plt.legend()
plt.grid(True)
plt.show()