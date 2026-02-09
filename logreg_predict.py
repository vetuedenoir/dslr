import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import sys
import argparse
import json
# from helper_and_class.LogisticRegression import LogisticRegression as lr


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
    numeric_df = df.drop(['Index', ' Hogwarts House', ' First Name', ' Last Name', ' Birthday', ' Best Hand'], axis='columns')

    if numeric_df.empty:
        print("Error: No numeric columns found in the dataset.")
        return []
    numeric_df = numeric_df.apply(pd.to_numeric, errors='coerce')
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1)
                              .astype(bool))

    # Find features with correlation greater than 0.98
    to_drop = [column for column in upper.columns if any(upper[column] > 0.98)]

    # Drop highly correlated features
    features = numeric_df.drop(columns=to_drop, errors='ignore')

    # Get the list of remaining features
    final_features = list(features.columns)

    # Remove specific features if they exist
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

    # Vérifie qu'il n'y a plus de NaN
    if data[features_cols].isnull().any().any():
        print("Warning: Some NaN values remain after filling with median.")
        data = data.dropna()  # Supprime les lignes restantes avec des NaN

    if data.empty:
        print("Error: No data left after processing.")
        sys.exit(1)

    x_raw = data[features_cols].to_numpy(dtype=float)

    # Ajout des caractéristiques polynomiales
    x_poly = np.hstack((x_raw, x_raw ** 2))

    # Standardisation (Z-Score)
    mean = np.mean(x_poly, axis=0)
    std = np.std(x_poly, axis=0)
    std[std == 0] = 1  # Évite la division par zéro
    x_final = (x_poly - mean) / std
    return x_final


def main():
    args = parse()
    thetas = load_thetas(args.weight)
    x = load_data(args.dataset)
    print(x)


if __name__ == "__main__":
    main()
