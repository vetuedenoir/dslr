import pandas as pd
import argparse
import sys
import numpy as np
from math import sqrt


class TinyStatistician():
    def __init__(self):
        pass

    @staticmethod
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
        return float(resulte)

    @staticmethod
    def median(array):
        if not isinstance(array, (list, set, tuple, np.ndarray)):
            return None
        if isinstance(array, (set, tuple)):
            array = list(array)
        lenght = len(array)
        if lenght == 0:
            return None
        if lenght == 1:
            return array[0]
        for element in array:
            if not isinstance(element, (int, float)):
                return None
        array.sort()
        if lenght % 2 != 0:
            return float(array[int((lenght / 2))])
        elif lenght >= 1:
            m1 = array[int(lenght / 2) - 1]
            m2 = array[int((lenght / 2))]
            return float((m1 + m2) / 2)
        return float(array[0])

    @staticmethod
    def quartile(array):
        if not isinstance(array, (list, set, tuple, np.ndarray)):
            return None
        if isinstance(array, (set, tuple)):
            array = list(array)
        lenght = len(array)
        if lenght == 0:
            return None
        if lenght == 1:
            return float(array[0])
        for element in array:
            if not isinstance(element, (int, float)):
                return None
        array.sort()

        quart = lenght / 4
        return [float(array[int(quart)]), float(array[int(quart * 3)])]

    @staticmethod
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

        if not isinstance(array, (list, set, tuple, np.ndarray)):
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

    @staticmethod
    def var(array):
        if not isinstance(array, (list, set, tuple, np.ndarray)):
            return None

        if isinstance(array, (set, tuple)):
            array = list(array)
        if len(array) == 0:
            return None
        if len(array) == 1:
            return float(array[0])
        m = TinyStatistician.mean(array)
        if m is None:
            return None
        ecart = [x - m for x in array]
        ecart_carre = [x * x for x in ecart]
        somme_carre = 0
        for x in ecart_carre:
            somme_carre += x
        result = somme_carre / (len(array) - 1)
        return float(result)

    @staticmethod
    def std(array):
        v = TinyStatistician.var(array)
        if v is None:
            return None
        return sqrt(v)

    @staticmethod
    def min(array):
        if not isinstance(array, (list, set, tuple, np.ndarray)):
            return None
        if isinstance(array, (set, tuple)):
            array = list(array)

        return float(min(array))

    @staticmethod
    def max(array):
        if not isinstance(array, (list, set, tuple, np.ndarray)):
            return None
        if isinstance(array, (set, tuple)):
            array = list(array)

        return float(max(array))


def parse():
    parser = argparse.ArgumentParser(prog="describe")
    parser.add_argument("data_file",
                        type=str,
                        help="The data_set file with extension csv")
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
    description["25%"] = TinyStatistician.percentile(data_feature, 25.0)
    description["50%"] = TinyStatistician.percentile(data_feature, 50.0)
    description["75%"] = TinyStatistician.percentile(data_feature, 75.0)
    description["Max"] = TinyStatistician.max(data_feature)
    return description


def process_features(data_frame):
    features_name = [
                    "Arithmancy",
                    "Astronomy",
                    "Herbology",
                    "Defense Against the Dark Arts",
                    "Divination",
                    "Muggle Studies",
                    "Ancient Runes",
                    "History of Magic",
                    "Transfiguration",
                    "Potions",
                    "Care of Magical Creatures",
                    "Charms",
                    "Flying"]

    infos_features = {}

    for f in features_name:
        feature = data_frame[f].dropna().to_numpy(dtype=float)
        infos_features[f] = describe_feature(feature)

    return infos_features


def display_information(info):
    features_name = list(info.keys())

    truncate_features_name = []
    for f in features_name:
        if len(f) > 13:
            truncate_features_name.append(f[:13] + '.')
        elif len(f) < 8:
            truncate_features_name.append(f + '\t')
        else:
            truncate_features_name.append(f)
    header = "\t" + "\t".join(truncate_features_name)
    print(header)

    stats_order = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]

    for stat in stats_order:
        line = stat
        for feature in features_name:
            line += f"\t{info[feature][stat]:5f}"
        print(line)


def main():
    args = parse()
    data_frame = read_csv(args.data_file)
    features_description = process_features(data_frame)
    display_information(features_description)


if __name__ == "__main__":
    main()
