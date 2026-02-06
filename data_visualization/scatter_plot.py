
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def put_data_in_df():
    """Function that open the .csv and load it into a DF"""
    df = pd.read_csv('../assets/dataset_train.csv')
    df = df.select_dtypes(include=np.number)
    df = df.drop("Index", axis="columns")
    return df


def find_correlation(df: pd.DataFrame):
    """Function that find the correlation and print it into a scatter plot"""
    df = df.corr()
    df = df.unstack()
    df = df.sort_values()
    plt.figure(figsize=(10, 8))
    plt.suptitle("Correlation Scratter")
    print(df)

    z = 1
    for i in range(1, 8, 2):
        plt.subplot(2, 2, z)
        keys = str(df.index[i])
        keys = keys.split("'")
        new_keys = []
        new_keys.append(keys[1])
        new_keys.append(keys[3])
        new_df = put_data_in_df()

        plt.xlabel(new_keys[0])
        plt.ylabel(new_keys[1])
        plt.scatter(new_df[new_keys[0]], new_df[new_keys[1]], c="blue")
        z += 1
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = put_data_in_df()
    find_correlation(df)
