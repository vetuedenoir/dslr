import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def print_histogram():
    """Function that print the histogram"""
    df = pd.read_csv("../assets/dataset_train.csv")
    df = df.drop(['Index', 'First Name', 'Last Name', 'Birthday', 'Best Hand'], axis='columns')
    num_df = df.select_dtypes(include=np.number)
    slytherin = df[df['Hogwarts House'] == 'Slytherin']
    gryffindor = df[df['Hogwarts House'] == 'Gryffindor']
    hufflepuff = df[df['Hogwarts House'] == 'Hufflepuff']
    ravenclaw = df[df['Hogwarts House'] == 'Ravenclaw']
    plt.figure(figsize=(18, 10))
    plt.suptitle("Histogram")

    z = 1
    for matiere in num_df:
        ax = plt.subplot(4, 4, z)
        plt.title(matiere)
        plt.hist(slytherin[matiere], bins=25, alpha=0.5, label="sly", color="purple")
        plt.hist(gryffindor[matiere], bins=25, alpha=0.5, label="gry", color="red")
        plt.hist(hufflepuff[matiere], bins=25, alpha=0.5, label="huff", color="blue")
        plt.hist(ravenclaw[matiere], bins=25, alpha=0.5, label="rav", color="green")
        plt.legend()
        z += 1
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print_histogram()