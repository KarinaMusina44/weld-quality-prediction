import json
import re
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import sklearn
import sys
assert sys.version_info >= (3, 5)
assert sklearn.__version__ >= "0.20"

with open("config.json", encoding="utf-8") as f:
    cfg = json.load(f)
    COLUMNS = cfg["columns"]
    DATA_PATH = cfg["data_path"]


def import_data():
    """Imports data from a CSV file into a pandas DataFrame with clean column names. 
    """
    try:
        data = pd.read_csv(DATA_PATH, sep=r"\s+", header=None)
        print(f"Data imported successfully from {file_path}")
        data.columns = COLUMNS
        return data
    except Exception as e:
        print(f"Error importing data: {e}")
        return None


def handle_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and format numeric data in a pandas DataFrame in the case of intervals or '<' patterns.

    Rules:
    - If a cell contains '<value', it is replaced by value / 2.
    - If a cell contains 'value1-value2', it is replaced by the average of the two values.
    - Otherwise, the cell remains unchanged.
    """

    def format_cell(cell):
        """Apply formatting rules to a single cell."""
        if isinstance(cell, str):
            cell = cell.strip()

            # Case 1: "<value" pattern
            if re.match(r'^<\s*\d+(\.\d+)?$', cell):
                value = float(cell.replace('<', '').strip())
                return value / 2

            # Case 2: "value1-value2" interval pattern
            if re.match(r'^\d+(\.\d+)?\s*-\s*\d+(\.\d+)?$', cell):
                value1, value2 = re.split(r'\s*-\s*', cell)
                return (float(value1) + float(value2)) / 2

        # Default: return the cell unchanged
        return cell

    # Apply the formatting function to all cells
    formatted_df = df.applymap(format_cell)

    return formatted_df


def handle_nitrogen(df: pd.DataFrame, nitrogen_column_name: str) -> pd.DataFrame:
    """
    Function to process the 'nitrogen' column in the DataFrame.
    If the value in the nitrogen column is a string, it converts it to a numeric value and 
    handles any conversion errors by setting invalid parsing to NaN.
    """
    if nitrogen_column_name in df.columns:
        df[nitrogen_column_name] = pd.to_numeric(
            df[nitrogen_column_name], errors='coerce')
    return df


def handle_hardness(df: pd.DataFrame, hardness_column_name: str) -> pd.DataFrame:
    """
    Function to process the 'hardness' column in the DataFrame.

    The column contains strings in the format "Value1(HvValue2)".
    This function extracts Value1 (the numeric part before the '(') and converts it to a numeric value.
    Non-numeric or malformed entries are set to NaN.
    """

    def extract_value(cell):
        """Extract the numeric value before '(' and convert it to float."""
        if isinstance(cell, str):
            # Use regex to find the number before '(' if it exists
            match = re.match(r'^\s*([0-9]+(?:\.[0-9]+)?)\s*\(', cell)
            if match:
                return float(match.group(1))
            # If no match, try converting the entire string directly
            try:
                return float(cell)
            except ValueError:
                return None
        return cell

    # Apply extraction to the specified column
    df[hardness_column_name] = df[hardness_column_name].apply(extract_value)

    # Convert the column to numeric, coercing invalid values to NaN
    df[hardness_column_name] = pd.to_numeric(
        df[hardness_column_name], errors='coerce')

    return df


def drop_non_informative_columns(df: pd.DataFrame) -> pd.DataFrame:

    df.drop('weld_id', axis=1, inplace=True)
    df.drop('col_45', axis=1, inplace=True)
    return df


def drop_not_chosen_target(df: pd.DataFrame, target_columns: list[str]) -> pd.DataFrame:
    """Drops all target columns except the one chosen for Machine Learning."""
    for col in df.columns:
        if col not in target_columns and col in df.columns:
            df.drop(col, axis=1, inplace=True)
    return df
