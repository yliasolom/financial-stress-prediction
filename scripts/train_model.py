"""
Script to train the financial stress prediction model and save artifacts
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 2025

def preprocess_credit_age(df):
    """Convert credit_age_months from string to numeric"""
    df['credit_age_months_numeric'] = df['credit_age_months'].apply(
        lambda x: int(x.replace(' y.', '').replace(' m.', '').split(' ')[0]) * 12 +
                  int(x.replace(' y.', '').replace(' m.', '').split(' ')[1])
        if pd.notna(x) else np.nan
    )
    df.drop('credit_age_months', axis=1, inplace=True)
    return df

def fix_negative_values(df):
    """Fix negative values in specific columns"""
    df['num_savings_accounts'] = df['num_savings_accounts'].apply(lambda x: max(x, 0))
    df['avg_loan_delay_days'] = df['avg_loan_delay_days'].apply(lambda x: max(x, 0))
    return df

def identify_outlier_columns(df, numerical_cols, threshold=1.5):
    """Identify columns with outliers using IQR method"""
    outlier_cols = []
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers_count = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col].count()
        if outliers_count > 0:
            outlier_cols.append(col)
    return outlier_cols

def main():
    print("Loading training data...")
    train_df = pd.read_csv('../data/raw/train.csv').drop('Unnamed: 0', axis=1)

    print(f"Training data shape: {train_df.shape}")
    print(f"Target distribution:\n{train_df['financial_stress_level'].value_counts()}")

    # Preprocessing
    print("\nPreprocessing data...")
    train_df = preprocess_credit_age(train_df)
    train_df = fix_negative_values(train_df)

    # Identify numerical and categorical columns
    numerical_cols = train_df.select_dtypes(include=np.number).columns.tolist()
    object_cols = train_df.select_dtypes(include='object').columns
    categorical_cols = object_cols.drop(['worker_id', 'financial_stress_level'], errors='ignore').tolist()

    # Identify outlier columns
    numerical_cols_outliers = identify_outlier_columns(train_df, numerical_cols)
    print(f"Columns with outliers: {len(numerical_cols_outliers)} out of {len(numerical_cols)}")

    # Calculate statistics for imputation
    train_medians = train_df[numerical_cols].median().to_dict()
    train_means = train_df[numerical_cols].mean().to_dict()

    # Fill missing values
    print("Filling missing values...")
    for col in numerical_cols:
        if col in numerical_cols_outliers:
            train_df[col].fillna(train_df[col].median(), inplace=True)
        else:
            train_df[col].fillna(train_df[col].mean(), inplace=True)

    for col in categorical_cols:
        train_df[col].fillna('Unknown', inplace=True)

    # Prepare features and target
    X = train_df.drop(['financial_stress_level', 'worker_id'], axis=1)
    y = train_df['financial_stress_level']

    # Encode target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
    )

    # Get feature columns for preprocessing
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include='object').columns.tolist()

    # Create preprocessor
    print("Creating preprocessor...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
        ]
    )

    # Fit preprocessor on training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

    # Train model
    print("\nTraining RandomForestClassifier...")
    print("Parameters: n_estimators=270, max_depth=35")

    clf = RandomForestClassifier(
        n_estimators=270,
        max_depth=35,
        class_weight=None,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    clf.fit(X_train_processed, y_train)

    # Validate
    y_pred = clf.predict(X_val_processed)
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')

    print(f"\nValidation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score (weighted): {f1:.4f}")

    # Train on full dataset
    print("\nTraining on full dataset...")
    X_full = train_df.drop(['financial_stress_level', 'worker_id'], axis=1)
    y_full = train_df['financial_stress_level']
    y_full_encoded = label_encoder.fit_transform(y_full)

    preprocessor_full = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
        ]
    )

    X_full_processed = preprocessor_full.fit_transform(X_full)

    clf_full = RandomForestClassifier(
        n_estimators=270,
        max_depth=35,
        class_weight=None,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    clf_full.fit(X_full_processed, y_full_encoded)

    # Save artifacts
    print("\nSaving model artifacts...")

    artifacts = {
        'model': clf_full,
        'preprocessor': preprocessor_full,
        'label_encoder': label_encoder,
        'train_medians': train_medians,
        'train_means': train_means,
        'numerical_cols_outliers': numerical_cols_outliers,
        'numerical_features': numerical_features,
        'categorical_features': categorical_features,
        'feature_names': X_full.columns.tolist()
    }

    joblib.dump(artifacts, '../models/model_artifacts.joblib')
    print("Model artifacts saved to ../models/model_artifacts.joblib")

    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()
