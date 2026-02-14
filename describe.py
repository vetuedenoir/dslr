import pandas as pd
import argparse
import sys
from scipy.stats import skew
from helper_and_class.TinyStatistician import TinyStatistician as TinyStat


def parse():
    """
    Parses command-line arguments for the describe script.

    Returns:
        argparse.Namespace: An object containing the path to the data file.
    """
    parser = argparse.ArgumentParser(prog="describe")
    parser.add_argument("data_file",
                        type=str,
                        help="The dataset file with extension csv")
    return parser.parse_args()


def read_csv(file: str):
    """
    Reads a CSV file and returns its contents as a pandas DataFrame.

    Args:
        file (str): Path to the CSV file.

    Returns:
        pd.DataFrame: The data read from the CSV file,
        or exits with an error if the file cannot be opened.
    """
    data = pd.read_csv(file)
    if data is None:
        print(f"Error: Cannot open the file: {file}")
        sys.exit(1)
    return data


def describe_feature(data_feature, length: int):
    """
    Describes a feature by calculating various statistics.

    Args:
        data_feature (pd.Series | np.ndarray): The feature data to describe.
        length (int): Total length of the dataset.

    Returns:
        dict: A dictionary containing the calculated statistics.
    """
    description = {}
    description["Count"] = len(data_feature)
    description["missing"] = float((length - len(data_feature)) / length)
    description["Mean"] = TinyStat.mean(data_feature)
    description["Std"] = TinyStat.std(data_feature)
    description["Var"] = TinyStat.var(data_feature)
    description["Min"] = TinyStat.min(data_feature)
    description["25%"] = TinyStat.percentile(data_feature, 25.0)
    description["50%"] = TinyStat.percentile(data_feature, 50.0)
    description["75%"] = TinyStat.percentile(data_feature, 75.0)
    description["IQR"] = description["75%"] - description["25%"]
    description["OUT<"] = description["25%"] - 1.5 * description["IQR"]
    description["OUT>"] = description["75%"] + 1.5 * description["IQR"]
    description["Max"] = TinyStat.max(data_feature)
    description["Skew"] = skew(data_feature)
    return description


def process_features(data_frame):
    """
    Processes each feature in the dataset and calculates statistics.

    Args:
        data_frame (pd.DataFrame): The dataset containing the features.

    Returns:
        dict: A dictionary containing the statistics for each feature.
    """
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

    features_info = {}

    for feature in features_name:
        feature_data = data_frame[feature].dropna().to_numpy(dtype=float)
        features_info[feature] = describe_feature(
                                            feature_data,
                                            len(data_frame[feature]))

    return features_info


def display_information(info):
    """
    Displays the descriptive statistics for each feature in a tabular format.

    Args:
        info (dict): A dictionary containing the descriptive statistics
        for each feature.
    """
    features_name = list(info.keys())

    truncated_features_name = []
    for feature in features_name:
        if len(feature) > 13:
            truncated_features_name.append(feature[:13] + '.')
        elif len(feature) < 8:
            truncated_features_name.append(feature + '\t')
        else:
            truncated_features_name.append(feature)
    header = "\t" + "\t".join(truncated_features_name)
    print(header)

    stats_order = ["Count", "missing", "Mean", "Std", "Var",
                   "Min", "OUT<", "25%", "50%", "75%", "IQR",
                   "Max", "OUT>", "Skew"]

    for stat in stats_order:
        line = stat
        for feature in features_name:
            if info[feature][stat] > 10000000:
                line += f"\t{info[feature][stat]:5e}"
            else:
                line += f"\t{info[feature][stat]:5f}"
        print(line)


def main():
    """
    Main function: Parses arguments, reads the dataset,
    processes features, and displays statistics.
    """
    try:
        args = parse()
        data_frame = read_csv(args.data_file)
        features_description = process_features(data_frame)
        display_information(features_description)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
