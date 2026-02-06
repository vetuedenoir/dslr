from LogisticRegression import LogisticRegression as lr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot(loss_vec: list):
    plt.plot(loss_vec)
    plt.ylabel('Loss evolution')
    plt.xlabel("number of iteration")
    plt.show()



# x = np.array([[1, 2, 3, 4, 5, -1, -2, -4, -5]])

# mylog = lr(theta=np.array([[1], [2], [3], [4], [5], [-1], [-2], [-3], [-4], [-5]]))

# sx = mylog.sigmoid_(x)
# print(sx)


# X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89., 144.]])
# Y = np.array([[0.98], [0.6], [0.85]])
# mylog = lr(theta=np.array([[1.], [1.], [1.], [1.], [1]]), algo='mini-batch_GD', batch_size=2)


# y_hat = mylog.log_predict_(X)
# print("Prediction avant entrainement: y_hat: \n", y_hat)

# print("\nLoss avant entrainement:\n", mylog.log_loss_(Y, y_hat), "\n")


# mylog.alpha = 1.6e-4
# mylog.max_iter = 20000
# mylog.fit_(X, Y)
# print("\nThetas :\n", mylog.theta)

# y_hat = mylog.log_predict_(X)
# print("\nPrediction: y_hat: \n", y_hat)

# print("\nLoss:\n", mylog.log_loss_(Y, y_hat), "\n")



X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [3., 5., 9., 14.]])
Y = np.array([[1], [0], [1]])
thetas = np.array([[2], [0.5], [7.1], [-4.3], [2.09]])
mylog = lr(thetas, algo='SGD', batch_size=2)
# Example 0:
y_hat = mylog.log_predict_(X)
print("Prediction avant entrainement: y_hat: \n", y_hat)

# Output:
# array([[0.99930437],
# [1.
# ],
# [1.
# ]])
# Example 1:
print("\nLoss avant entrainement:\n", mylog.log_loss_(Y, y_hat), "\n")
# Output:
# 11.513157421577002
# Example 2:
mylog.fit_(X, Y)
print("\nThetas :\n", mylog.theta)
# Output:
# array([[ 2.11826435]
# [ 0.10154334]
# [ 6.43942899]
# [-5.10817488]
# [ 0.6212541 ]])
# Example 3:
y_hat = mylog.log_predict_(X)
print("\nPrediction: y_hat: \n", y_hat)
# Output:
# array([[0.57606717]
# [0.68599807]
# [0.06562156]])
# Example 4:
print("\nLoss:\n", mylog.log_loss_(Y, y_hat), "\n")
# Output:
# 1.4779126923052268

plot(mylog.historique)





