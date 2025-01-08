# created a heat map to visualize correlation between features
# Quickly highlights strong relationships between features.
# Identifies redundant features that might not add value to your model.

import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # Visualization library on top of matplotlib
from utils import is_valid_csv

def main():
    args = sys.argv[1:]
    if len(args) != 1:
        print("Usage: python bonus_visualize.py <path/to/dataset.csv>")
        exit(1)
    path_to_dataset = args[0]
    if not is_valid_csv(path_to_dataset):
        exit(1)
    df = pd.read_csv(path_to_dataset)

    # Select numerical columns for correlation matrix
    numerical_columns = [
        "Arithmancy",
        "Astronomy",
        "Herbology",
        "Defense Against the Dark Arts",
        "Divination",
        "Muggle Studies",
        "Ancient Runes",
        "History of Magic",
        "Transfiguration",
    ]

    # Calculate the correlation matrix
    correlation_matrix = df[numerical_columns].corr()

    # Create the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        correlation_matrix,
        annot=True, 
        cmap="coolwarm", 
        fmt=".2f", 
        linewidths=0.5
    )
    plt.title("Feature Correlation Heatmap", fontsize=16)
    plt.show()

if __name__ == "__main__":
    main()
