import numpy as np
import pandas as pd
import sys
import argparse
import json
import csv

from helper_and_class.LogisticRegression import LogisticRegression as lr


def parse():
    parser = argparse.ArgumentParser(prog="logreg_predict")
    parser.add_argument("dataset", type=str, help="the dataset to evaluate")
    parser.add_argument("weight", type=str,
                        help="The weihgts needed by the model")

    return parser.parse_args()


def load_thetas(path: str):
    try:
        with open(path, 'r') as file:
            return {house: tethas for house, tethas in json.load(file).items()}
    except Exception as e:
        print(f"Error: cannot read in file {e}")
        sys.exit(1)


def to_keep(df: pd.DataFrame) -> list:
    """A function that return the best courses to train the model """
    numeric_df = df.drop([
                        'Index',
                        'Hogwarts House', 'First Name',
                        'Last Name', 'Birthday',
                        'Best Hand'], axis='columns')
    if numeric_df.empty:
        print("Error: No numeric columns found in the dataset.")
        return []
    numeric_df = numeric_df.apply(pd.to_numeric, errors='coerce')
    corr_matrix = numeric_df.corr().abs()

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1)
                              .astype(bool))

    to_drop = [column for column in upper.columns if any(upper[column] > 0.98)]

    features = numeric_df.drop(columns=to_drop, errors='ignore')

    final_features = list(features.columns)

    if 'Arithmancy' in final_features:
        final_features.remove('Arithmancy')
    if 'Care of Magical Creatures' in final_features:
        final_features.remove('Care of Magical Creatures')

    return final_features


def load_data(path: str):
    data = pd.read_csv(path)
    if data.empty:
        print("Error: The dataset is empty.")
        sys.exit(1)

    courses = to_keep(data)
    features_cols = courses
    for course in features_cols:
        if course not in data.columns:
            print(f"Warning: Column '{course}' not found in the dataset.")
            continue
        data[course] = pd.to_numeric(data[course], errors='coerce')
        data[course] = data[course].fillna(data[course].median())

    if data[features_cols].isnull().any().any():
        print("Warning: Some NaN values remain after filling with median.")
        data = data.dropna()
    if data.empty:
        print("Error: No data left after processing.")
        sys.exit(1)

    x_raw = data[features_cols].to_numpy(dtype=float)
    x_poly = np.hstack((x_raw, x_raw ** 2))

    mean = np.mean(x_poly, axis=0)
    std = np.std(x_poly, axis=0)
    std[std == 0] = 1
    x_final = (x_poly - mean) / std
    return x_final


def prediction(x: np.ndarray, thetas: dict):
    """
    Predict Hogwarts house labels for each sample in x
        using logistic regression models.
    Parameters:
        x : np.ndarray, feature matrix of shape (n_samples, n_features).
        thetas : dict, dictionary mapping house names
            to their trained parameter vectors.

    Returns:
        house_prediction: Dictionary mapping sample indices (as strings)
            to predicted house names.
    """
    model_Gryf = lr(theta=np.array(thetas["Gryffindor"]))
    model_Huff = lr(theta=np.array(thetas["Hufflepuff"]))
    model_Slyt = lr(theta=np.array(thetas["Slytherin"]))
    model_Rave = lr(theta=np.array(thetas["Ravenclaw"]))
    pred_Gryf = model_Gryf.log_predict_(x)
    pred_Huff = model_Huff.log_predict_(x)
    pred_Slyt = model_Slyt.log_predict_(x)
    pred_Rave = model_Rave.log_predict_(x)

    house_prediction = {}
    for yG, yH, yS, yR, i in zip(
            pred_Gryf,
            pred_Huff,
            pred_Slyt,
            pred_Rave,
            range(len(pred_Gryf))
            ):
        yBest = max([yG, yH, yS, yR])
        if yBest == yG:
            house_prediction[f"{i}"] = "Gryffindor"
        elif yBest == yH:
            house_prediction[f"{i}"] = "Hufflepuff"
        elif yBest == yS:
            house_prediction[f"{i}"] = "Slytherin"
        elif yBest == yR:
            house_prediction[f"{i}"] = "Ravenclaw"

    return house_prediction


def save_prediction(prediction: dict):
    """
    Save house predictions to a CSV file named 'houses.csv'.

    Parameters:
        prediction : dict, dictionary containing prediction results
            with keys 'Index' and 'Hogwarts House'.
    """

    try:
        with open("houses.csv", 'w') as file:
            fieldnames = ['Index', 'Hogwarts House']
            writer = csv.writer(file)
            writer.writerow(fieldnames)
            for k, value in prediction.items():
                writer.writerow([k, value])
    except Exception as e:
        print(f"Cannot save the prediction in house.csv: {e}")


def main():
    args = parse()
    try:
        thetas = load_thetas(args.weight)
        x = load_data(args.dataset)
        house_pred = prediction(x, thetas)
        save_prediction(house_pred)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
