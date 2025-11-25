"""
Data preprocessing functions
"""
import pandas as pd
import numpy as np
from typing import Dict, List


def preprocess_credit_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert credit_age_months from string format (e.g., '17 y. 11 m.') to numeric

    Args:
        df: DataFrame with credit_age_months column

    Returns:
        DataFrame with credit_age_months_numeric column
    """
    df = df.copy()

    def convert_credit_age(x):
        if pd.isna(x):
            return np.nan
        try:
            parts = x.replace(' y.', '').replace(' m.', '').split(' ')
            years = int(parts[0])
            months = int(parts[1])
            return years * 12 + months
        except:
            return np.nan

    df['credit_age_months_numeric'] = df['credit_age_months'].apply(convert_credit_age)
    df.drop('credit_age_months', axis=1, inplace=True)

    return df


def fix_negative_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix negative values in specific columns by clipping to 0

    Args:
        df: DataFrame with numerical columns

    Returns:
        DataFrame with corrected values
    """
    df = df.copy()

    if 'num_savings_accounts' in df.columns:
        df['num_savings_accounts'] = df['num_savings_accounts'].clip(lower=0)

    if 'avg_loan_delay_days' in df.columns:
        df['avg_loan_delay_days'] = df['avg_loan_delay_days'].clip(lower=0)

    return df


def fill_missing_values(
    df: pd.DataFrame,
    numerical_cols: List[str],
    categorical_cols: List[str],
    train_medians: Dict[str, float],
    train_means: Dict[str, float],
    numerical_cols_outliers: List[str]
) -> pd.DataFrame:
    """
    Fill missing values using training statistics

    Args:
        df: DataFrame with missing values
        numerical_cols: List of numerical column names
        categorical_cols: List of categorical column names
        train_medians: Dictionary of median values from training data
        train_means: Dictionary of mean values from training data
        numerical_cols_outliers: List of columns with outliers (use median)

    Returns:
        DataFrame with filled missing values
    """
    df = df.copy()

    # Fill numerical columns
    for col in numerical_cols:
        if col in df.columns and df[col].isnull().sum() > 0:
            if col in numerical_cols_outliers:
                fill_value = train_medians.get(col, df[col].median())
            else:
                fill_value = train_means.get(col, df[col].mean())
            df[col].fillna(fill_value, inplace=True)

    # Fill categorical columns
    for col in categorical_cols:
        if col in df.columns and df[col].isnull().sum() > 0:
            df[col].fillna('Unknown', inplace=True)

    return df


def preprocess_input_data(
    data: pd.DataFrame,
    train_medians: Dict[str, float],
    train_means: Dict[str, float],
    numerical_cols_outliers: List[str],
    numerical_features: List[str],
    categorical_features: List[str]
) -> pd.DataFrame:
    """
    Complete preprocessing pipeline for input data

    Args:
        data: Raw input DataFrame
        train_medians: Median values from training data
        train_means: Mean values from training data
        numerical_cols_outliers: Columns with outliers
        numerical_features: List of numerical feature names
        categorical_features: List of categorical feature names

    Returns:
        Preprocessed DataFrame ready for model prediction
    """
    df = data.copy()

    # Apply transformations
    df = preprocess_credit_age(df)
    df = fix_negative_values(df)
    df = fill_missing_values(
        df,
        numerical_features,
        categorical_features,
        train_medians,
        train_means,
        numerical_cols_outliers
    )

    return df
