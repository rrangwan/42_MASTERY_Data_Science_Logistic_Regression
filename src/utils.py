import os
import pandas as pd

def is_valid_csv(file_path):
    """Check if the file is a valid .csv file, exists, is not empty, and is readable."""
    if not os.path.isfile(file_path):
        print(f"Error: {file_path} does not exist or is not a file.")
        return False
    if not file_path.endswith('.csv'):
        print(f"Error: {file_path} is not a valid .csv file.")
        return False
    if os.path.getsize(file_path) == 0:
        print(f"Error: {file_path} is empty.")
        return False
    try:
        # Try reading the file using pandas to ensure it's a valid CSV
        pd.read_csv(file_path)
    except Exception as e:
        print(f"Error: Cannot read {file_path}. {e}")
        return False
    return True