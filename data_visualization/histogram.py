import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys


def print_histogram(path_dataset: str):
    """Function that print the histogram"""
    df = pd.read_csv(path_dataset)
    if df is None:
        print(f"Error: Cannot open the file: {path_dataset}")
        sys.exit(1)

    df = df.drop(['Index', 'First Name', 'Last Name',
                 'Birthday', 'Best Hand'], axis='columns')
    num_df = df.select_dtypes(include=np.number)
    slytherin = df[df['Hogwarts House'] == 'Slytherin']
    gryffindor = df[df['Hogwarts House'] == 'Gryffindor']
    hufflepuff = df[df['Hogwarts House'] == 'Hufflepuff']
    ravenclaw = df[df['Hogwarts House'] == 'Ravenclaw']
    plt.figure(figsize=(18, 10))
    plt.suptitle("Histogram")

    z = 1
    for matiere in num_df:
        plt.subplot(4, 4, z)
        plt.title(matiere)
        plt.hist(
            slytherin[matiere],
            bins=25,
            alpha=0.5,
            label="sly",
            color="purple")
        plt.hist(
            gryffindor[matiere],
            bins=25,
            alpha=0.5,
            label="gry",
            color="red")
        plt.hist(
            hufflepuff[matiere],
            bins=25,
            alpha=0.5,
            label="huff",
            color="blue")
        plt.hist(
            ravenclaw[matiere],
            bins=25,
            alpha=0.5,
            label="rav",
            color="green")
        plt.legend()
        z += 1
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="histogram")
    parser.add_argument(
        "dataset",
        type=str,
        help="the dataset")
    args = parser.parse_args()
    print_histogram(args.dataset)
