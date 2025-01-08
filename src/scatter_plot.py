import sys
import pandas as pd


from utils import is_valid_csv 
import matplotlib.pyplot as plt
import seaborn as sns #visualisation lib on top of matplotlib



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
        "Potions",
        "Care of Magical Creatures",
        "Charms",
        "Flying"
    ]

    # Create a scatter plot matrix
    fig, axes = plt.subplots(len(attributes), len(attributes), figsize=(16, 16))
    for row_idx, row_feature in enumerate(attributes):
        for col_idx, col_feature in enumerate(attributes):
            if row_feature == col_feature:
                axes[row_idx][col_idx].axis('off')  # Hide diagonal plots (optional)
                continue
            sns.scatterplot(
                data=df,
                x=col_feature,
                y=row_feature,
                hue="Hogwarts House",
                palette=house_colors,
                ax=axes[row_idx][col_idx],
                legend=False
            )

    # Adjust layout
    plt.tight_layout(h_pad=2, w_pad=2)  # Add padding between subplots
    plt.show()
    

if __name__ == "__main__":
    main()
