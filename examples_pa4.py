import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


def function(x):
    return 1 / (1 + math.pow(math.e, -x))


y = np.vectorize(function)
x = np.arange(-10, 10, 0.1)
fig, ax = plt.subplots()
ax.plot(x, y(x))
ax.grid(True)
ax.axvline(x=0, color='k')
ax.set_xlabel("x")
ax.set_ylabel("y")

plt.show()
