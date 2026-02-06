import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

from LogisticRegression import LogisticRegression as lr
from load_data import load_data

def parse():
    parser = argparse.ArgumentParser(prog="logreg_train")
    parser.add_argument("dataset", type=str, help="the dataset to train the model")
    parser.add_argument("-p", "--plot", action='store_true', help="plot the evolution of loss function")
    parser.add_argument("-a", "--algo", type=str, help="the algorithms to minimize the error")
    return parser.parse_args()

def plot(loss_vec: list, title: str):
    plt.plot(loss_vec)
    plt.ylabel('Loss evolution')
    plt.xlabel("number of iteration")
    plt.title(title)
    plt.show()

def train_model(x: np.ndarray , y: np.ndarray, thetas: np.ndarray, algo: str, p: bool, house_name: str):
    model = lr(thetas.copy(), max_iter=300, alpha=0.01, algo=algo)
    model.fit_(x, y)
    print(f"{house_name} loss final : {model.log_loss_(y, model.log_predict_(x))}")

    if p == True:
        plot(model.historique, house_name)
    return model

def save_thetas(models_thetas: list):
    house = ["Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin"]
    houses_thetas = {}
    houses_thetas["Gryffindor"] = models_thetas[0]
    houses_thetas["Ravenclaw"] = models_thetas[1]
    houses_thetas["Hufflepuff"] = models_thetas[2]
    houses_thetas["Slytherin"] = models_thetas[3]

    with open("assets/tethas.csv", "a") as file:
        for house in houses_thetas:
            line = house + "\t"
            for theta in houses_thetas[house]:
                line += f", {theta}"
            file.write(line + "\n")


def create_model(path : str, algo: str, plot: bool):
    thetas = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])

    x, huff_y, gryf_y, sly_y, rav_y = load_data(path)
    house = ["Gryffindor", "Hufflepuff", "Slytherin", "Ravenclaw"]

    with ProcessPoolExecutor() as executor:
        lr_gryf, lr_huff, lr_sly, lr_rav = executor.map(
            train_model,
            repeat(x),
            [gryf_y, huff_y, sly_y, rav_y],
            repeat(thetas),
            repeat(algo),
            repeat(plot),
            house
    )

    save_thetas([lr_gryf.theta, lr_rav.theta, lr_huff.theta, lr_sly.theta])

def main():
    args = parse()
    if args.algo is None:
        args.algo = "gradient_descent"
    try:
        create_model(args.dataset, args.algo, args.plot)     
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

