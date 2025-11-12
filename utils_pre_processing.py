import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
import re
import json
import seaborn as sns
from typing import Tuple
import sys
assert sys.version_info >= (3, 5)
assert sklearn.__version__ >= "0.20"


def import_data(file_path, config_path="config.json"):
    """Imports data from a CSV file into a pandas DataFrame using external config."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        data = pd.read_csv(file_path, header=None)
        print(f"Data imported successfully from {file_path}")

        data = data[0].str.split(
            r'\s+', expand=True).rename(columns=lambda i: f"col_{i+1}")
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
            print(
                f"⚠️  Kept '{col}' as non-numeric (contains non-convertible values).")
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
        df[nitrogen_column_name] = pd.to_numeric(
            df[nitrogen_column_name], errors='coerce')
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
    df[['hardness_kgmm2', 'hardness_kgmm2_flag']
       ] = df['hardness_kgmm2'].apply(extract_values)

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
    print("Dropped columns:")
    print(cols_to_drop)
    df = df.drop(columns=cols_to_drop)

    # Columns with 40-80% missing for review
    cols_to_review = missing_pct[(missing_pct >= 40) & (missing_pct <= 80)]
    if not cols_to_review.empty:
        print("Columns with 40-80% missing values (consider reviewing):")
        print(cols_to_review)

    return df


def correlation_matrix(processed_df: pd.DataFrame, target_col: str) -> None:
    """Generates and displays a correlation matrix heatmap for the DataFrame."""
    numeric_df = processed_df.select_dtypes(include=[np.number])
    if target_col in numeric_df.columns:
        ordered_cols = [target_col] + \
            [c for c in numeric_df.columns if c != target_col]
        numeric_df = numeric_df[ordered_cols]
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


def add_features(df):
    """Adds feature engineering columns for predicting yield strength"""

    # Carbon Equivalent (CE)
    df['ce_iww'] = (
        df['carbon_concentration_weight'] +
        df['manganese_concentration_weight'] / 6 +
        (df['chromium_concentration_weight'] + df['molybdenum_concentration_weight'] + df['vanadium_concentration_weight']) / 5 +
        (df['nickel_concentration_weight'] +
         df['copper_concentration_weight']) / 15
    )

    # Carbon Squared (C²)
    df['carbon_squared'] = df['carbon_concentration_weight'] ** 2

    # Mn/C Ratio
    df['mn_c_ratio'] = df['manganese_concentration_weight'] / \
        (df['carbon_concentration_weight'] + 1e-6)

    # Arc Energy (Voltage × Current)
    df['arc_energy'] = df['voltage_v'] * df['current_a']

    # HAZ Hardness Estimate
    df['haz_hardness'] = (
        90 +
        1050 * df['carbon_concentration_weight'] +
        45 * df['silicon_concentration_weight'] +
        30 * df['manganese_concentration_weight'] +
        5 * df['heat_input_kjmm1']
    )

    # Mn/S Ratio
    df['mn_s_ratio'] = df['manganese_concentration_weight'] / \
        (df['sulphur_concentration_weight'] + 1e-6)

    # Austenite Stabilizer
    df['austenite_stabilizer'] = df['manganese_concentration_weight'] / \
        2 + 10*df['carbon_concentration_weight'] + \
        df['nickel_concentration_weight']

    return df


def split_data(df, target='yield_strength_mpa', test_size=0.2, random_state=42):
    """
    Split the DataFrame into training and test sets using an 80/20 split.
    """
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def viz_outliers(X_train):
    num_cols = X_train.select_dtypes('number').columns.tolist()
    k = len(num_cols)
    ncols = 3
    nrows = (k + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(
        ncols*4.2, nrows*3.1), squeeze=False)
    axes = axes.ravel()

    for i, col in enumerate(num_cols):
        ax = axes[i]
        s = X_train[col]
        q005, q995 = s.quantile([0.005, 0.995])

        tails = s.le(q005) | s.ge(q995)
        mid = ~tails

        ax.scatter(X_train.index[mid], s[mid], s=8,
                   alpha=0.7, label='Central 99%')
        ax.scatter(X_train.index[tails], s[tails], s=10, alpha=0.85, color='red',
                   label='Tails (≤0.5% / ≥99.5%)')

        ax.axhline(q005, linestyle='--', linewidth=1)
        ax.axhline(q995, linestyle='--', linewidth=1)
        ax.set_title(col, fontsize=9)
        ax.tick_params(axis='x', labelrotation=45)

    # remove unused axes
    for j in range(i + 1, nrows * ncols):
        fig.delaxes(axes[j])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center',  frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def winsorize_columns(X_train, X_test, cols, lower=0.005, upper=0.995):
    """
    Fit quantile caps on X_train[cols] and apply the same caps to X_train and X_test.
    Only the provided `cols` are clipped; other columns are left unchanged.
    """

    q = X_train[cols].quantile([lower, upper])
    low, high = q.loc[lower], q.loc[upper]

    Xtr = X_train.copy()
    Xte = X_test.copy()
    Xtr[cols] = X_train[cols].clip(low, high, axis=1)
    Xte[cols] = X_test[cols].clip(low, high, axis=1)

    return Xtr, Xte


def one_hot_train_test(X_train_no_outlier: pd.DataFrame, X_test_no_outlier: pd.DataFrame):

    cat_cols = X_train_no_outlier.select_dtypes(
        include=["object", "category"]).columns.tolist()
    print(f'Categorical columns {cat_cols}')
    num_cols = [c for c in X_train_no_outlier.columns if c not in cat_cols]
    print(f'Numerical columns {num_cols}')

    ohe = OneHotEncoder(handle_unknown="ignore",
                        sparse_output=False, drop=None)

    ohe.fit(X_train_no_outlier[cat_cols])

    Xtr_cat = pd.DataFrame(
        ohe.transform(X_train_no_outlier[cat_cols]),
        index=X_train_no_outlier.index,
        columns=ohe.get_feature_names_out(cat_cols),
    )
    Xte_cat = pd.DataFrame(
        ohe.transform(X_test_no_outlier[cat_cols]),
        index=X_test_no_outlier.index,
        columns=ohe.get_feature_names_out(cat_cols),
    )

    Xtr_num = X_train_no_outlier[num_cols].copy()
    Xte_num = X_test_no_outlier[num_cols].copy()

    X_train_oh = pd.concat([Xtr_num, Xtr_cat], axis=1)
    X_test_oh = pd.concat([Xte_num, Xte_cat], axis=1)
    print(f'Columns after one-hot: {X_train_oh.columns.tolist()}')

    return X_train_oh, X_test_oh


def custom_impute_data(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Custom imputer implementing the specified rules with direct SimpleImputer calls (no helper).
    Assumes all listed columns exist in both train and test.
    """
    Xtr, Xte = X_train.copy(), X_test.copy()

    # Column groups
    mean_cols = [
        "sulphur_concentration_weight", "phosphorus_concentration_weight",
        "oxygen_concentration_ppm", "mn_s_ratio"
    ]
    zero_cols = [
        "nickel_concentration_weight", "chromium_concentration_weight",
        "molybdenum_concentration_weight", "vanadium_concentration_weight",
        "copper_concentration_weight", "aluminium_concentration_ppm",
        "boron_concentration_ppm", "niobium_concentration_ppm"
    ]
    median_cols = ["titanium_concentration_ppm", "nitrogen_concentration_ppm"]
    mode_cols = ["current_a", "voltage_v", "arc_energy",
                 "post_weld_heat_treatment_temperature_c", "post_weld_heat_treatment_time_h"]

    target_cols = [
        "ultimate_tensile_strength_mpa", "elongation_percent",
        "reduction_of_area_percent", "charpy_temperature_c",
        "charpy_impact_toughness_j"
    ]

    to_drop_after = ["ce_iww", "austenite_stabilizer"]

    # Mean
    imp_mean = SimpleImputer(strategy="mean")
    Xtr_mean = imp_mean.fit_transform(Xtr[mean_cols])
    Xte_mean = imp_mean.transform(Xte[mean_cols])
    Xtr[mean_cols] = pd.DataFrame(Xtr_mean, index=Xtr.index, columns=mean_cols)
    Xte[mean_cols] = pd.DataFrame(Xte_mean, index=Xte.index, columns=mean_cols)

    # Median
    imp_median = SimpleImputer(strategy="median")
    Xtr_median = imp_median.fit_transform(Xtr[median_cols])
    Xte_median = imp_median.transform(Xte[median_cols])
    Xtr[median_cols] = pd.DataFrame(
        Xtr_median, index=Xtr.index, columns=median_cols)
    Xte[median_cols] = pd.DataFrame(
        Xte_median, index=Xte.index, columns=median_cols)

    # Mode (most frequent)
    imp_mode = SimpleImputer(strategy="most_frequent")
    Xtr_mode = imp_mode.fit_transform(Xtr[mode_cols])
    Xte_mode = imp_mode.transform(Xte[mode_cols])
    Xtr[mode_cols] = pd.DataFrame(Xtr_mode, index=Xtr.index, columns=mode_cols)
    Xte[mode_cols] = pd.DataFrame(Xte_mode, index=Xte.index, columns=mode_cols)

    # Zero for rare alloying elements
    Xtr[zero_cols] = Xtr[zero_cols].fillna(0.0)
    Xte[zero_cols] = Xte[zero_cols].fillna(0.0)

    # For each target y, regress y on the remaining targets (median-filled predictors) and predict missing.
    train_medians = Xtr[target_cols].median()
    Xtr_pred = Xtr[target_cols].copy().fillna(train_medians)
    Xte_pred = Xte[target_cols].copy().fillna(train_medians)

    for ycol in target_cols:
        other = [c for c in target_cols if c != ycol]

        # rows where y is observed in train
        mask = Xtr[ycol].notna()
        if mask.sum() >= 2 and len(other) > 0:
            model = LinearRegression()
            model.fit(Xtr_pred.loc[mask, other], Xtr.loc[mask, ycol])

            # Predict missing in train
            miss_tr = Xtr[ycol].isna()
            if miss_tr.any():
                Xtr.loc[miss_tr, ycol] = model.predict(
                    Xtr_pred.loc[miss_tr, other])

            # Predict missing in test
            miss_te = Xte[ycol].isna()
            if miss_te.any():
                Xte.loc[miss_te, ycol] = model.predict(
                    Xte_pred.loc[miss_te, other])

    # Drop custom features
    Xtr.drop(columns=to_drop_after, inplace=True)
    Xte.drop(columns=to_drop_after, inplace=True)

    # Quick log/check
    n_tr = int(Xtr.isna().sum().sum())
    n_te = int(Xte.isna().sum().sum())
    print(
        f" Remaining NaN — train: {n_tr}, test: {n_te}.")

    return Xtr, Xte


def viz_imputed(X_before: pd.DataFrame,
                X_after: pd.DataFrame,
                ncols: int = 3) -> None:
    """
    Scatter grid per numeric feature.
    Blue = observed (not imputed), Red = imputed (positions that were NaN in X_before).
    """
    # Use only numeric columns present in both frames
    num_cols = X_after.select_dtypes(include="number").columns
    cols = [c for c in num_cols if c in X_before.columns]

    k = len(cols)
    nrows = (k + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(
        ncols*4.2, nrows*3.1), squeeze=False)
    axes = axes.ravel()

    # Imputed mask: True where the original value was NaN
    imp_mask = X_before.isna()

    for i, col in enumerate(cols):
        ax = axes[i]
        s = X_after[col]
        idx = X_after.index

        obs = ~imp_mask[col]
        imp = imp_mask[col]

        # Blue = base (observed), Red = imputed
        ax.scatter(idx[obs], s[obs], s=8, alpha=0.7,
                   color="blue", label="Base (observed)")
        if imp.any():
            ax.scatter(idx[imp], s[imp], s=12, alpha=0.9,
                       color="red", label="Imputed")

        n_imp = int(imp.sum())
        ax.set_title(f"{col}  (imputed: {n_imp})", fontsize=9)
        ax.tick_params(axis="x", labelrotation=45)

    # Remove unused axes
    for j in range(i + 1, nrows * ncols):
        fig.delaxes(axes[j])

    # Shared legend
    handles, labels = [], []
    for ax in axes[:k]:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)
    if handles:
        fig.legend(handles, labels, loc="upper center", frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def scaling_data(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame]:
    """Scales the features in the DataFrame using StandardScaler."""

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(
        X_train_scaled, index=X_train.index, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(
        X_test_scaled,  index=X_test.index,  columns=X_test.columns)
    return X_train_scaled, X_test_scaled


def collinearity_management(df_train: pd.DataFrame, df_test: pd.DataFrame, threshold: float = 0.9) -> tuple[pd.DataFrame]:
    """
    Removes highly collinear features from the DataFrames based on the specified threshold.
    
    Parameters
    ----------
    df_train : pd.DataFrame
        Training set.
    df_test : pd.DataFrame
        Test set.
    threshold : float, optional
        Correlation threshold above which features are considered collinear, by default 0.9.
    
    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, list[str]]
        Reduced training and test sets, and the list of removed columns.
    """
    corr_matrix = df_train.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper_triangle.columns if any(upper_triangle[col] > threshold)]

    if len(to_drop) == len(df_train.columns):
        raise ValueError("All features are collinear above the threshold. Adjust the threshold or review the data.")

    df_train_reduced = df_train.drop(columns=to_drop)
    df_test_reduced = df_test.drop(columns=to_drop, errors='ignore')

    print(f"Removed {len(to_drop)} features due to high collinearity (>{threshold}): {to_drop}")

    return df_train_reduced, df_test_reduced, to_drop


def pca(X_train: pd.DataFrame, X_test: pd.DataFrame, n_components: int) -> tuple[pd.DataFrame]:
    """
    Applies Principal Component Analysis (PCA) to reduce the dimensionality of the feature set.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training set of numerical features.
    X_test : pd.DataFrame
        Test set of numerical features.
    n_components : int
        Number of principal components to retain.
    
    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, PCA]
        PCA-transformed training and test sets, and the fitted PCA model.
    """
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    X_train_pca_df = pd.DataFrame(
        X_train_pca, index=X_train.index, columns=[f'PC{i+1}' for i in range(n_components)]
    )
    X_test_pca_df = pd.DataFrame(
        X_test_pca, index=X_test.index, columns=[f'PC{i+1}' for i in range(n_components)]
    )

    explained = pca.explained_variance_ratio_.sum() * 100
    print(f"PCA retained {n_components} components explaining {explained:.2f}% of total variance.")

    return X_train_pca_df, X_test_pca_df, pca