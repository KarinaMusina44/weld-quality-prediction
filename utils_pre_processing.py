import sys
assert sys.version_info >= (3, 5)
import sklearn
assert sklearn.__version__ >= "0.20"
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import re
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


import json

def import_data(file_path, config_path="config.json"):
    """Imports data from a CSV file into a pandas DataFrame using external config."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        data = pd.read_csv(file_path, header=None)
        print(f"Data imported successfully from {file_path}")

        data = data[0].str.split(r'\s+', expand=True).rename(columns=lambda i: f"col_{i+1}")
        data.replace('N', None, inplace=True)
        # Use config for renaming and dropping columns
        data = data.rename(columns=config["rename_columns"])

        for col in config.get("drop_columns", []):
            if col in data.columns:
                data.drop(col, axis=1, inplace=True)

        return data

    except Exception as e:
        print(f"Error importing data: {e}")
        return None


def safe_numeric_conversion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Try to convert each column to numeric.
    If the conversion fails (because of non-numeric values),
    the original column is preserved.
    """
    converted_df = df.copy()
    for col in converted_df.columns:
        try:
            # Try to convert entire column to numeric (handles None automatically)
            converted_col = pd.to_numeric(converted_df[col], errors='raise')
            converted_df[col] = converted_col
            print(f"✅ Converted '{col}' to numeric.")
        except Exception:
            print(f"⚠️  Kept '{col}' as non-numeric (contains non-convertible values).")
    return converted_df

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

def handle_hardness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and extracts data from the 'hardness_kgmm2' column.
    
    Steps:
    1. Extracts the first numeric value found in the string (before parentheses).
    2. Extracts the numeric value that follows 'Hv' (e.g., in Hv30 → 30).
    3. Creates a new column 'hardness_kgmm2_flag' containing the Hv value.
    4. Replaces the original column with the first numeric value as a float.
    
    Parameters:
        df (pd.DataFrame): Input dataframe containing the column 'hardness_kgmm2'.
    
    Returns:
        pd.DataFrame: Updated dataframe with cleaned 'hardness_kgmm2' and new 'hardness_kgmm2_flag' columns.
    """
    
    # Define regex patterns
    first_number_pattern = re.compile(r'(\d+(?:\.\d+)?)')
    hv_flag_pattern = re.compile(r'Hv(\d+(?:\.\d+)?)')
    
    def extract_values(value):
        if pd.isna(value):
            return pd.Series([None, None])
        
        text = str(value)
        first_number = None
        hv_flag = None
        
        # Extract the first numeric value
        match_first = re.search(r'(\d+(?:\.\d+)?)', text)
        if match_first:
            first_number = float(match_first.group(1))
        
        # Extract the number after 'Hv'
        match_hv = hv_flag_pattern.search(text)
        if match_hv:
            hv_flag = float(match_hv.group(1))
        
        return pd.Series([first_number, hv_flag])
    
    # Apply extraction to the dataframe
    df[['hardness_kgmm2', 'hardness_kgmm2_flag']] = df['hardness_kgmm2'].apply(extract_values)
    
    return df



def drop_non_informative_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes non-informative columns based on the proportion of missing values.
    
    Steps:
    1. Drop columns with more than 80% missing values.
    2. Print columns with 40% to 80% missing values for further examination.
    
    Parameters:
        df (pd.DataFrame): Input dataframe.
    
    Returns:
        pd.DataFrame: Dataframe with non-informative columns removed.
    """
    
    # Calculate percentage of missing values per column
    missing_pct = df.isna().mean() * 100
    
    # Columns to drop (>80% missing)
    cols_to_drop = missing_pct[missing_pct > 80].index.tolist()
    df = df.drop(columns=cols_to_drop)
    
    # Columns with 40-80% missing for review
    cols_to_review = missing_pct[(missing_pct >= 40) & (missing_pct <= 80)]
    if not cols_to_review.empty:
        print("Columns with 40-80% missing values (consider reviewing):")
        print(cols_to_review)
    
    return df


def correlation_matrix(processed_df: pd.DataFrame) -> None:
    """Generates and displays a correlation matrix heatmap for the DataFrame."""
    numeric_df = processed_df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    print("Numeric columns used for correlation:", list(numeric_df.columns))
    print(corr)

    # optional nicer plot using seaborn (import is safe here)

    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", vmin=-1, vmax=1, center=0,
                linewidths=0.5, cbar_kws={"shrink": 0.6})
    plt.title("Correlation matrix (numeric columns)")
    plt.tight_layout()
    plt.show()



def drop_not_chosen_target(df: pd.DataFrame, threshold: float = 0.1):
    """
    Supprime les colonnes numériques peu corrélées avec les autres colonnes.
    
    Paramètres :
    - df : DataFrame pandas à nettoyer
    - threshold : valeur minimale de corrélation absolue avec au moins une autre colonne pour conserver la colonne
    
    Retour :
    - cleaned_df : DataFrame nettoyé
    - dropped_cols : liste des colonnes supprimées
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().abs()
    
    # Pour chaque colonne, calculer la corrélation maximale avec les autres colonnes
    max_corr = corr_matrix.apply(lambda col: col.drop(col.name).max(), axis=0)
    
    # Colonnes à garder : celles qui ont au moins une corrélation >= threshold avec une autre colonne
    keep_cols = max_corr[max_corr >= threshold].index.tolist()
    
    # Colonnes supprimées
    dropped_cols = [col for col in numeric_df.columns if col not in keep_cols]
    
    # Créer le DataFrame nettoyé
    cleaned_df = df.drop(columns=dropped_cols)
    
    return cleaned_df, dropped_cols