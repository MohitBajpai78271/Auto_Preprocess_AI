import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

def preprocess_data(df, drop_cols=None, num_strategy="median", cat_strategy="most_frequent", apply_scaling=True, apply_encoding=True):
    summary = []

    # Drop user-selected columns
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)
        summary.append(f"Dropped columns: {drop_cols}")

    # Track nulls
    null_counts = df.isnull().sum()
    summary.append(f"Initial null values per column:\n{null_counts[null_counts > 0].to_dict()}")

    # Separate columns
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Impute numerics
    if num_cols:
        num_imputer = SimpleImputer(strategy=num_strategy)
        df[num_cols] = num_imputer.fit_transform(df[num_cols])
        summary.append(f"Imputed numerical columns using strategy: {num_strategy}")

    # Impute categoricals
    if cat_cols:
        if cat_strategy == "drop":
            df.dropna(subset=cat_cols, inplace=True)
            summary.append("Dropped rows with nulls in categorical columns")
        else:
            cat_imputer = SimpleImputer(strategy=cat_strategy)
            df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
            summary.append(f"Imputed categorical columns using strategy: {cat_strategy}")

    # One-hot encode
    if apply_encoding and cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        summary.append("One-hot encoded categorical columns")

    # Scale numerics
    if apply_scaling and num_cols:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        summary.append("Applied standard scaling to numerical columns")

    # Detect and remove outliers (IsolationForest)
    try:
        iso = IsolationForest(contamination=0.01, random_state=42)
        outliers = iso.fit_predict(df)
        df['outlier'] = outliers
        df = df[df['outlier'] == 1].drop(columns='outlier')
        summary.append("Removed outliers using IsolationForest (1% contamination).")
    except Exception as e:
        summary.append(f"Outlier detection skipped due to error: {str(e)}")

    return df, summary
