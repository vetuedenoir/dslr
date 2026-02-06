import numpy as np
import pandas as pd
from data_visualization.pair_plot import to_keep

def load_data():
    """load the data, keep only the courses with heterogene repartition of data
    return numpy arrays, to the model"""
    data = pd.read_csv("assets/dataset_train.csv")

    courses = to_keep()
    data_num = data[courses]
    data_num = data_num.drop(['Hogwarts House'], axis="columns")
    #attention pas sur
    data_num = data_num.dropna()
    
    huff_y = (data['Hogwarts House'] == 'Hufflepuff').to_numpy(dtype=int)
    gryf_y = (data['Hogwarts House'] == 'Gryffindor').to_numpy(dtype=int)
    sly_y = (data['Hogwarts House'] == 'Slytherin').to_numpy(dtype=int)
    rav_y = (data['Hogwarts House'] == 'Ravenclaw').to_numpy(dtype=int)

    return data, huff_y, gryf_y, sly_y,rav_y


if __name__ == "__main__":
    load_data()