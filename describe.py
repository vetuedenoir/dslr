import pandas as pd
import argparse
import sys
import numpy as np
from math import sqrt


class   TinyStatistician():
    def __init__(self):
        pass
    
    staticmethod
    def mean(array):
        if not isinstance(array, (list, set, tuple, np.ndarray)):
            print("Not the good type")
            return None
        if len(array) == 0:
            print("len == 0")
            return None
        resulte = 0.0
        for element in array:
            if not isinstance(element, (int, float)):
                print("Les element de sont pas des float")
                return None
            resulte += element
        resulte = resulte / len(array)
        return resulte
    

    staticmethod
    def median(array):
        if not isinstance(array, (list, set, tuple)):
            return None
        if isinstance(array, (set, tuple)):
            array = list(array)
        l = len(array)
        if l == 0:
            return None
        if l == 1:
            return array[0]
        for element in  array:
            if not isinstance(element, (int, float)):
                return None
        array.sort()
        if l % 2 != 0:
            return float(array[int((l / 2))])
        elif l >= 1:
            return float((array[int(l / 2) - 1] + array[int((l / 2) )]) / 2)
        return float(array[0])
    

    staticmethod
    def quartile(array):
        if not isinstance(array, (list, set, tuple)):
                return None
        if isinstance(array, (set, tuple)):
            array = list(array)
        l = len(array)
        if l == 0:
            return None
        if l == 1:
            return float(array[0])
        for element in  array:
            if not isinstance(element, (int, float)):
                return None
        array.sort()
        # first, third = 0.0
        quart = l / 4
        return [float(array[int(quart)]), float(array[int(quart * 3)])]


    staticmethod
    def percentile(array, p):
        """
        Calcule le pᵉ percentile d'une liste de valeurs,
        en utilisant une interpolation linéaire.

        Args:
            array (list | set | tuple): Données numériques.
            p (float): Percentile à calculer (0 <= p <= 100).

        Returns:
            float | None: La valeur du percentile, ou None si entrée invalide.
        """

        if not isinstance(array, (list, set, tuple)):
            return None

        if isinstance(array, (set, tuple)):
            array = list(array)

        if len(array) == 0:
            return None
        if not (0 <= p <= 100):
            return None

        array = sorted(array)
        n = len(array)
        if n == 1:
            return float(array[0])
        if p == 0:
            return float(array[0])
        if p == 100:
            return float(array[-1])

        #  Rang exact (position fractionnaire dans la liste triée)
        rank = (p / 100) * (n - 1)
        lower_index = int(rank)
        upper_index = lower_index + 1
        fraction = rank - lower_index

        #  Interpolation linéaire
        lower_value = array[lower_index]
        upper_value = array[upper_index]
        result = lower_value + fraction * (upper_value - lower_value)

        return float(result)
    

    staticmethod
    def var(array):
        if not isinstance(array, (list, set, tuple)):
            return None

        if isinstance(array, (set, tuple)):
            array = list(array)
        if len(array) == 0:
            return None
        if len(array) == 1:
            return float(array[0])
        mean = mean(array)
        if mean == None:
            return None
        ecart = [x - mean for x in array]
        ecart_carre = [x * x for x in ecart]
        somme_carre = 0
        for x in ecart_carre:
            somme_carre += x
        result = somme_carre / (len(array) - 1)
        return float(result)
    

    staticmethod
    def std(array):
        v = TinyStatistician.var(array)
        if v == None:
            return None
        return sqrt(v)
    
    staticmethod
    def min(array):
        if not isinstance(array, (list, set, tuple)):
            return None
        if isinstance(array, (set, tuple)):
            array = list(array)
        
        return min(array)
    
    staticmethod
    def max(array):
        if not isinstance(array, (list, set, tuple)):
            return None
        if isinstance(array, (set, tuple)):
            array = list(array)
        
        return max(array)




def parse():
    parser = argparse.ArgumentParser(prog="describe")
    parser.add_argument("data_file", type=str, help="The data_set file with extension csv")
    return parser.parse_args()


def read_csv(file: str):
    data = pd.read_csv(file)
    if data is None:
        print(f"Error: Cannot open the file: {file}")
        sys.exit(1)
    return data


def describe_feature(data_feature):
    description = {}


    description["Count"] = len(data_feature)
    description["Mean"] = TinyStatistician.mean(data_feature)
    description["Std"] = TinyStatistician.std(data_feature)
    description["Min"] = TinyStatistician.min(data_feature)
    description["25%"] = TinyStatistician.quartile(data_feature)
    description["50%"] = TinyStatistician.percentile(data_feature, 50.0)
    description["75%"] = TinyStatistician.percentile(data_feature, 75.0)
    description["Max"] = TinyStatistician.max(data_feature)

    return description



def process_features(data):
    features = ["Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts", "Divination", "Muggle Studies",
                "Ancient Runes", "History of Magic", "Transfiguration", "Potions", "Care of Magical Creatures", "Charms", "Flying"]
    
    description_off_all_features = {}

    for f in  features:
        description_off_all_features[f] = describe_feature(data[f].to_numpy(dtype=float).astype(np.float32))
    
    return description_off_all_features


def display_information(info):
    names_of_feature = "\t"

    for t in info:
        names_of_feature += f"\t  {t}"
    print(names_of_feature)
    for t in info:
        print(t, "\t\t", info[t])
       

def main():
    args = parse()
    data_tab = read_csv(args.data_file)
    features_description =  process_features(data_tab)
    display_information(features_description)


if __name__ == "__main__":
    main()
