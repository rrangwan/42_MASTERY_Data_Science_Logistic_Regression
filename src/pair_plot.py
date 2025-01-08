import sys
import pandas as pd


from utils import is_valid_csv 
import matplotlib.pyplot as plt
import seaborn as sns #visualisation lib on top of matplotlib

# Look for feature pairs where points from different classes (e.g., represented by different colors) show a clear separation. This indicates that these features have predictive power for distinguishing between classes.

# Not care of magical creatures and potions feature

def main():
    args = sys.argv[1:]
    if len(args) != 1:
        print("Usage: python scatter_plot.py <path/to/dataset.csv>")
        exit(1)
    path_to_dataset = args[0]
    if is_valid_csv(path_to_dataset) == False:
        exit(1)
    df = pd.read_csv(path_to_dataset)

    house_colors = {
    'Gryffindor': 'red',
    'Slytherin': 'green',
    'Ravenclaw': 'blue',
    'Hufflepuff': 'yellow'
    }

    attributes = [
        "Hogwarts House",
        "Potions",
        "Care of Magical Creatures",
        "Charms",
        "Flying"
    ]

    df = df[attributes]
    sns.pairplot(data=df, hue="Hogwarts House", palette=house_colors)
    plt.show()
    

if __name__ == "__main__":
    main()
