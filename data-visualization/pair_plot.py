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
    df = df[df.index.get_level_values(0) < df.index.get_level_values(1)]
    df = df[df > 0.3]
    # Tri, on garde que les valeurs qui ne sont pas trop homogenes
    to_keep = list(df.index.get_level_values(0)) + list(df.index.get_level_values(1))
    to_keep = list(set(to_keep)) + ["Hogwarts House"]
    # Nouveau tableau clean
    new_df = pd.read_csv("../assets/dataset_train.csv")
    new_df = new_df[to_keep]

    sb.pairplot(new_df, hue="Hogwarts House")
    plt.show()

if __name__ == "__main__":
    pair_plot()