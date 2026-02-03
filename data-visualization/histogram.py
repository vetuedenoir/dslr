import pandas as pd
import numpy as np

def print_histogram():
    df = pd.read_csv("../assets/dataset_train.csv")
    print(df)
    num_df = df.select_dtypes(include=np.number)
    print(num_df)

if __name__ == "__main__":
    print_histogram()