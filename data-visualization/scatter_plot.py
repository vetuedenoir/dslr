
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np

def put_data_in_df():
    df = pd.read_csv('../assets/dataset_train.csv')
    df = df.select_dtypes(include=np.number)
    df = df.drop("Index", axis="columns")
    return df

def find_correlation(df : pd.DataFrame):
    df = df.corr()
    df = df.unstack()
    df = df.sort_values()

    keys = str(df.index[0])
    keys = keys.split("'")
    new_keys = []
    new_keys.append(keys[1])
    new_keys.append(keys[3])
    new_df = put_data_in_df()
    plt.title("Correlation Scratter")
    plt.xlabel(new_keys[0])
    plt.ylabel(new_keys[1])
    plt.scatter(new_df[new_keys[0]], new_df[new_keys[1]], c="red")
    plt.show()


if __name__ == "__main__":
    df = put_data_in_df()
    find_correlation(df)