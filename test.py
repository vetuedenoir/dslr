from LogisticRegression import LogisticRegression as lr
import numpy as np

x = np.array([[1, 2, 3, 4, 5, -1, -2, -4, -5]])

mylog = lr(theta=np.array([[1], [2], [3], [4], [5], [-1], [-2], [-3], [-4], [-5]]))

sx = mylog.sigmoid_(x)
print(sx)