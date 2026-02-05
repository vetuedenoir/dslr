import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import pandas as pd


def pair_plot():
    """A function that print """
    df = pd.read_csv("../assets/dataset_train.csv")
    df = df.drop(["Index", "First Name", "Last Name", "Hogwarts House", "Birthday", "Best Hand"], axis="columns")
    # Exclure les valeurs homogenes
    df = df.corr()
    df = df.stack()
    df = df.sort_values(ascending=True)
    for i in range(len(df)):
        index1 = df[i]
        print(index1)
    # df = df.drop
    #pairplot avec seaborn
    print(df)


if __name__ == "__main__":
    pair_plot()