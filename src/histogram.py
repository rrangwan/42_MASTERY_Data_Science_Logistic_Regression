import sys
import pandas as pd


from utils import is_valid_csv 
import matplotlib.pyplot as plt
import seaborn as sns #visualisation lib on top of matplotlib



def main():
    args = sys.argv[1:]
    if len(args) != 1:
        print("Usage: python histogram.py <path/to/dataset.csv>")
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

    # Determine grid layout
    cols_per_row = 3
    num_rows = -(-len(attributes) // cols_per_row)  

    figure, axes = plt.subplots(nrows=num_rows, ncols=cols_per_row, figsize=(16, 4 * num_rows))
    axes = axes.flatten()  

    figure.suptitle("Hogwarts Feature Distributions", fontsize=12)

    # Create histograms
    for idx, attribute in enumerate(attributes):
        sns.histplot(
            data=df,
            x=attribute,
            hue="Hogwarts House",
            palette=house_colors,
            ax=axes[idx]
        )
        axes[idx].set_title(attribute)

    # Hide any unused subplots
    for ax in axes[len(attributes):]:
        ax.axis('off')

    # Adjust spacing between rows
    plt.tight_layout(h_pad=3) 
    plt.show()

    

if __name__ == "__main__":
    main()
