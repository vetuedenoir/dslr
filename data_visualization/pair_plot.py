import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import pandas as pd


def to_keep(path: str) -> list:
    """A function that return the best courses to train the model """
    df = pd.read_csv(path)
    numeric_df = df.select_dtypes(include=[np.number])

    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(
        np.triu(
            np.ones(
                corr_matrix.shape),
            k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.98)]

    features = numeric_df.drop(columns=to_drop)

    final_features = list(features.columns)

    if 'Arithmancy' in final_features:
        final_features.remove('Arithmancy')
    if 'Care of Magical Creatures' in final_features:
        final_features.remove('Care of Magical Creatures')

    final_features.append("Hogwarts House")
    return final_features


def pair_plot(path: str):
    """A function that print the pair plot to choose the data"""
    df = pd.read_csv(path)
    df = df.drop(["Index",
                  "First Name",
                  "Last Name",
                  "Hogwarts House",
                  "Birthday",
                  "Best Hand"],
                 axis="columns")

    # Exclure les valeurs homogenes
    df = df.corr()
    df = df.stack()
    df = df.sort_values(ascending=True)
    df = df[df.index.get_level_values(0) < df.index.get_level_values(1)]
    df = df[df > 0.3]
    # Tri, on garde que les valeurs qui ne sont pas trop homogenes
    to_keep = list(df.index.get_level_values(0)) + \
        list(df.index.get_level_values(1))
    to_keep = list(set(to_keep)) + ["Hogwarts House"]
    # Nouveau tableau clean
    new_df = pd.read_csv(path)
    new_df = new_df[to_keep]

    sb.pairplot(new_df, hue="Hogwarts House")
    plt.show()


if __name__ == "__main__":
    pair_plot()
