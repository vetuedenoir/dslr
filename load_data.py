import numpy as np
import pandas as pd
from data_visualization.pair_plot import to_keep

def load_data(path : str):
    """load the data, keep only the courses with heterogene repartition of data
    return numpy arrays, to the model"""
    data = pd.read_csv(path)

    courses = to_keep(path)
    data = data[courses].dropna()
    data_num = data[courses].dropna()

    courses.remove("Hogwarts House")
    data_num = data_num.drop(['Hogwarts House'], axis="columns")

    data_num = data_num.to_numpy(dtype=float)
    
    min_val = np.min(data_num, axis=0)
    max_val = np.max(data_num, axis=0)
    new_data_num = (data_num - min_val) / (max_val - min_val)
    print(new_data_num)
    print(np.max(new_data_num, axis=0))
    
    huff_y = (data['Hogwarts House'] == 'Hufflepuff').dropna().to_numpy(dtype=float).reshape(data_num.shape[0], 1)
    gryf_y = (data['Hogwarts House'] == 'Gryffindor').dropna().to_numpy(dtype=float).reshape(data_num.shape[0], 1)
    sly_y = (data['Hogwarts House'] == 'Slytherin').dropna().to_numpy(dtype=float).reshape(data_num.shape[0], 1)
    rav_y = (data['Hogwarts House'] == 'Ravenclaw').dropna().to_numpy(dtype=float).reshape(data_num.shape[0], 1)

    return data_num, huff_y, gryf_y, sly_y, rav_y


if __name__ == "__main__":
    load_data()