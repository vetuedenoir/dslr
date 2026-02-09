import numpy as np
import matplotlib.pyplot as plt

import argparse
import json

from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

from helper_and_class.LogisticRegression import LogisticRegression as lr
from helper_and_class.load_data import load_data

# from sklearn.metrics import accuracy_score


ITERATION = 1000
ALPHA = 0.08
BATCH_SIZE = 256


def parse():
    """
    Parses command-line arguments for model training.

    Returns:
        argparse.Namespace: An object containing the dataset path, plot flag,
        and optimization algorithm.
    """

    parser = argparse.ArgumentParser(prog="logreg_train")
    parser.add_argument(
        "dataset",
        type=str,
        help="the dataset to train the model")
    parser.add_argument(
        "-p",
        "--plot",
        action='store_true',
        help="plot the evolution of loss function")
    parser.add_argument(
        "-a",
        "--algo",
        type=str,
        help="the algorithms to minimize the error")
    return parser.parse_args()


def plot(loss_vec: list, title: str):
    """
    Plots the evolution of the loss over iterations.

    Args:
        loss_vec (list): List of loss values at each iteration.
        title (str): Title of the plot.
    """

    plt.plot(loss_vec)
    plt.ylabel('Loss evolution')
    plt.xlabel("number of iteration")
    plt.title(title)
    plt.show()


def train_model(x: np.ndarray, y: np.ndarray, thetas: np.ndarray,
                algo: str, p: bool, house_name: str):
    """
    Trains a logistic regression model for a given house.

    Args:
        x (np.ndarray): Features of the dataset.
        y (np.ndarray): Labels of the dataset.
        thetas (np.ndarray): Initial model parameters.
        algo (str): Optimization algorithm to use.
        p (bool): If True, plot the loss evolution.
        house_name (str): Name of the house for which the model
            is trained.

    Returns:
        model: The trained model.
    """

    model = lr(thetas.copy(),
               max_iter=ITERATION,
               alpha=ALPHA,
               algo=algo,
               batch_size=BATCH_SIZE)
    model.fit_(x, y)

    y_hat = model.log_predict_(x)
    print(f"{house_name} loss final : {model.log_loss_(y, y_hat)}")

    # y_hat = (y_hat >= 0.5).astype(float)
    # score = accuracy_score(y, y_hat)
    # print(score)

    if p is True:
        plot(model.historique, house_name)

    return model


def save_thetas(models_thetas: dict):
    """
    Saves model parameters (thetas) to a JSON file.

    Args:
        models_thetas (dict): Dictionary mapping each house
            to its parameters (thetas).
    """
    try:
        file = "assets/tethas.json"
        with open(file, "w") as file:
            json.dump(models_thetas, file, indent=4)
    except IOError as e:
        print(f"Error: cannot write in file: {e}")


def create_model(path: str, algo: str, plot: bool):
    """
    Creates and trains a model for each house, then saves the parameters.

    Args:
        path (str): Path to the dataset.
        algo (str): Optimization algorithm to use.
        plot (bool): If True, plot the loss evolution.
    """

    x, huff_y, gryf_y, sly_y, rav_y = load_data(path)
    n_features = x.shape[1]
    thetas = np.zeros((n_features + 1, 1))
    house_data = {
        "Gryffindor": gryf_y,
        "Hufflepuff": huff_y,
        "Slytherin": sly_y,
        "Ravenclaw": rav_y
    }

    with ProcessPoolExecutor() as executor:
        results = executor.map(
            train_model,
            repeat(x),
            house_data.values(),
            repeat(thetas),
            repeat(algo),
            repeat(plot),
            house_data.keys()
        )

    models_thetas = {
                    house: result.theta.tolist()
                    for house, result
                    in zip(house_data.keys(), results)
                    }
    save_thetas(models_thetas)


def main():
    """
    Main function: Parses arguments and starts the model training process.
    """
    args = parse()
    if args.algo is None:
        args.algo = "gradient_descent"
    try:
        create_model(args.dataset, args.algo, args.plot)
        print("Training finished!")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
