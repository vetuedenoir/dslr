import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import argparse

from LogisticRegression import LogisticRegression as lr
from load_data import load_data

def parse():
    parser = argparse.ArgumentParser(prog="logreg_train")
    parser.add_argument("dataset", type=str, help="the dataset to train the model")
    parser.add_argument("-p", "--plot", help="plot the evolution of loss function")
    parser.add_argument("-a", "--algo", help="the algorithms to minimize the error")
    parser.add_argument("-o", "--opti", help="the optimization algorithms")
    return parser.parse_args()

def plot(loss_vec: list):
    plt.plot(loss_vec)
    plt.ylabel('Loss evolution')
    plt.xlabel("number of iteration")
    plt.show()


def create_model(path : str):
    thetas = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])

    x, huff_y, gryf_y, sly_y, rav_y = load_data(path)

    lr_gryf = lr(thetas, max_iter=5000, alpha=0.01)
    lr_gryf.fit_(x, gryf_y)
    plot(lr_gryf.historique)

    # lr_huff = lr(thetas, max_iter=30000)
    # lr_huff.fit_(x, huff_y)

    # lr_sly = lr(thetas, max_iter=30000)
    # lr_sly.fit_(x, sly_y)

    # lr_rav = lr(thetas, max_iter=30000)
    # lr_rav.fit_(x, rav_y)

    # with open("assets/tethas.csv", "w") as file:
    #     file.write("Ravenclaw : " + lr_rav.theta)

def main():
    args = parse()
    try:
        create_model(args.dataset)
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

