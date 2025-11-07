import sys
assert sys.version_info >= (3, 5)
import sklearn
assert sklearn.__version__ >= "0.20"
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import re


def import_data(file_path):
    """Imports data from a CSV file into a pandas DataFrame. 
        Clean the columns names by removing spaces, converting to lowercase.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Data imported successfully from {file_path}")
        data = data[0].str.split(r'\s+', expand=True).rename(columns=lambda i: f"col_{i+1}")
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
        df[nitrogen_column_name] = pd.to_numeric(df[nitrogen_column_name], errors='coerce')
    return df

import pandas as pd
import re


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
    df[hardness_column_name] = pd.to_numeric(df[hardness_column_name], errors='coerce')

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

def add_features(df):
    """Adds feature engineering columns for predicting yield strength"""

    # Carbon Equivalent (CE)
    df['ce_iww'] = (
        df['carbon'] +
        df['manganese'] / 6 +
        (df['chromium'] + df['molybdenum'] + df['vanadium']) / 5 +
        (df['nickel'] + df['copper']) / 15
    )

    # Carbon Squared (C²)
    df['carbon_squared'] = df['carbon'] ** 2

    # Mn/C Ratio
    df['mn_c_ratio'] = df['manganese'] / (df['carbon'] + 1e-6)

    # Arc Energy (Voltage × Current)
    df['arc_energy'] = df['voltage'] * df['current']

    # HAZ Hardness Estimate
    df['haz_hardness'] = (
        90 +
        1050 * df['carbon'] +
        45 * df['silicon'] +
        30 * df['manganese'] +
        5 * df['heat_input']
    )

    # Mn/S Ratio
    df['mn_s_ratio'] = df['manganese'] / (df['sulphur'] + 1e-6)

    # Austenite Stabilizer 
    df['austenite_stabilizer'] = df['manganese']/2 + 10*df['carbon'] + df['nickel']

    return df


from sklearn.model_selection import train_test_split

def split_data(df, target='yield_strength', test_size=0.2, random_state=42):
    """
    Split the DataFrame into training and test sets using an 80/20 split.
    """
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test