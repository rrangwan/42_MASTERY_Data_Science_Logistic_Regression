import sys

import numpy as np
import pandas as pd
from prettytable import PrettyTable

from utils import is_valid_csv

def Describe(df: pd.DataFrame) -> PrettyTable:
    """
    Generate descriptive statistics 
    """
    # Find the first numerical column (starting from 'Potions') and the last numerical column (ending at 'Flying')
    start_col = 'Potions'
    end_col = 'Flying'
    
    # Filter the DataFrame to select columns
    numeric_df = df.loc[:, start_col:end_col].select_dtypes(include=[np.number])

    # Custom statistical functions
    def custom_count(series):
        return len([x for x in series if not pd.isna(x)])

    def custom_mean(series):
        valid_values = [x for x in series if not pd.isna(x)]
        return sum(valid_values) / len(valid_values) if valid_values else 0

    def custom_std(series):
        valid_values = [x for x in series if not pd.isna(x)]
        if len(valid_values) < 2:
            return 0
        mean = custom_mean(series)
        variance = sum((x - mean) ** 2 for x in valid_values) / (len(valid_values) - 1)
        return variance ** 0.5

    def custom_min(series):
        valid_values = [x for x in series if not pd.isna(x)]
        return min(valid_values) if valid_values else float('inf')

    def custom_max(series):
        valid_values = [x for x in series if not pd.isna(x)]
        return max(valid_values) if valid_values else float('-inf')

    def custom_quantile(series, q):
        valid_values = sorted([x for x in series if not pd.isna(x)])
        if not valid_values:
            return 0
        n = len(valid_values)
        pos = q * (n - 1)
        i = int(pos)
        f = pos - i
        if i + 1 >= n:
            return valid_values[i]
        return valid_values[i] + f * (valid_values[i + 1] - valid_values[i])

    # Calculate statistics for each column
    stats = {}
    for col in numeric_df.columns:
        series = numeric_df[col]
        stats[col] = {
            "Count": custom_count(series),
            "Mean": custom_mean(series),
            "Std": custom_std(series),
            "Min": custom_min(series),
            "25%": custom_quantile(series, 0.25),
            "50%": custom_quantile(series, 0.50),
            "75%": custom_quantile(series, 0.75),
            "Max": custom_max(series),
            "Bonnus NaN": series.isnull().sum()
        }

    # Create a table with "Statistic" as the first column and feature names as the headers
    table = PrettyTable()
    table.field_names = ["Statistic"] + list(numeric_df.columns)

    # Function to format values to 5 significant figures
    def format_value(val):
        if pd.isna(val) or val == float('inf') or val == float('-inf'):
            return "NaN"
        return f"{val:.5g}"
    
    # Add the statistics as rows
    stat_names = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max", "Bonnus NaN"]
    for stat_name in stat_names:
        row = [stat_name] + [format_value(stats[col][stat_name]) for col in numeric_df.columns]
        table.add_row(row)

    # Adjust column alignment and set width for better readability
    for col in table.field_names:
        table.align[col] = "r"
        table.max_width[col] = 15

    table.padding_width = 1
    
    return table

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