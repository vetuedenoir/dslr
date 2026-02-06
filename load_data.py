import numpy as np
import pandas as pd
from data_visualization.pair_plot import to_keep

def load_data(path : str):
    """
    Load data, impute missing values, add polynomial features, 
    standardize, and return X and One-vs-All targets.
    """
    data = pd.read_csv(path)
    courses = to_keep(path)
    
    features_cols = courses[:-1] 
    for course in features_cols:
        data[course] = data[course].fillna(data.groupby('Hogwarts House')[course].transform('median'))

    data = data[courses].dropna()
    
    x_raw = data[features_cols].to_numpy(dtype=float)
    
    # (Polynomial)
    x_poly = np.hstack((x_raw, x_raw ** 2))
    
    #Standardisation (Z-Score)
    mean = np.mean(x_poly, axis=0)
    std = np.std(x_poly, axis=0)
    std[std == 0] = 1 # Sécurité division par zéro
    x_final = (x_poly - mean) / std
    
    houses = data['Hogwarts House']
    huff_y = (houses == 'Hufflepuff').to_numpy(dtype=float).reshape(-1, 1)
    gryf_y = (houses == 'Gryffindor').to_numpy(dtype=float).reshape(-1, 1)
    sly_y = (houses == 'Slytherin').to_numpy(dtype=float).reshape(-1, 1)
    rav_y = (houses == 'Ravenclaw').to_numpy(dtype=float).reshape(-1, 1)

    return x_final, huff_y, gryf_y, sly_y, rav_y


if __name__ == "__main__":
    load_data()