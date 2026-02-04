import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import argparse

import LogisticRegression as lr


def load_data(file: str) -> pd.DataFrame:
    """
    Load the data_set and return the interisting features as pd.DataFrame
    """
    df = pd.read_csv(file)
    if df is None:
        print(f"Cannot open the file: {file}")
        sys.exit(1)
    #charger les features interessant.
    #
    #
    return df
    
    #le seul optimizer facile a mettre en place c'est adam
    #A voir si on le met par default 

def parse():
    parser = argparse.ArgumentParser(prog="logreg_train")
    parser.add_argument("dataset", type=str, help="the dataset to train the model")
    parser.add_argument("-p", "--plot", help="plot the evolution of loss function")
    parser.add_argument("-a", "--algo", help="the algorithms to minimize the error")
    parser.add_argument("-o", "--opti", help="the optimization algorithms")
    return parser.parse_args()


def main():
    args = parse()
    try:
        df = load_data(args.dataset)
        print(df)
        print(df.describe())
        #comme c'est un multi-classifier il faudra faire plusieur logistic regression

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

