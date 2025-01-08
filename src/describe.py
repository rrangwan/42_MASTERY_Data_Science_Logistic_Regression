import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd
from prettytable import PrettyTable

from utils import is_valid_csv # Imran, you can use this function to validate the csv file

# Imran, you can redo below to not use built in statistic funtions
# I have added a bonus field of  Nan values, dont delete it 
def Describe(df: pd.DataFrame) -> PrettyTable:
    """
    Generate descriptive statistics for the numerical columns in a dataset, starting from the first numerical field.
    :param df: A pandas DataFrame
    :return: A PrettyTable containing the descriptive statistics
    """
    # Find the first numerical column (starting from 'Potions') and the last numerical column (ending at 'Flying') to get the features
    start_col = 'Potions'
    end_col = 'Flying'
    
    # Filter the DataFrame to select columns
    numeric_df = df.loc[:, start_col:end_col].select_dtypes(include=[np.number])

    # Calculate statistics for each column
    statistics = {
        "Count": numeric_df.count(),
        "Mean": numeric_df.mean(),
        "Std": numeric_df.std(),
        "Min": numeric_df.min(),
        "25%": numeric_df.quantile(0.25),
        "50%": numeric_df.quantile(0.50),
        "75%": numeric_df.quantile(0.75),
        "Max": numeric_df.max(),
        "Bonnus NaN": numeric_df.isnull().sum(),
    }

    # Create a table with "Statistic" as the first column and feature names as the headers
    table = PrettyTable()
    table.field_names = ["Statistic"] + list(numeric_df.columns)

    # Function to format values to 5 significant figures
    def format_value(val):
        if pd.isna(val):
            return "NaN"
        return f"{val:.5g}"
    
     # Add the statistics as rows, with the statistic name in the first column
    for stat_name, stat_values in statistics.items():
        formatted_values = [format_value(val) for val in stat_values]
        table.add_row([stat_name] + formatted_values)

     # Adjust column alignment and set width for better readability
    for col in table.field_names:
        table.align[col] = "r"  # Right-align the columns for numeric data
        table.max_width[col] = 15  # Set a reasonable maximum column width

    # Enable table to have equal column width for aesthetics
    table.padding_width = 1  # Add padding between columns to make it easier on the eyes

    
    return table



# Imran, keep the main below as is
def main():
    args = sys.argv[1:]
    if len(args) != 1:
        print("Usage: python describe.py <path/to/dataset.csv>")
        exit(1)
    path_to_dataset = args[0]
    if is_valid_csv(path_to_dataset) == False:
        exit(1)
    df = pd.read_csv(path_to_dataset)
    result = Describe(df)
    print(result)
    

if __name__ == "__main__":
    main()
