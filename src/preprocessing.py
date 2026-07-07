import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

def preprocess_pipeline():
    print("Starting preprocessing pipeline...")
    
    # 1. Load dataset
    filepath = os.path.join('dataset', 'credit_card.csv')
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"Loaded dataset containing {len(df)} rows and {len(df.columns)} columns.")
    
    # 2. Handle missing values
    # For numeric features, fill with median
    numeric_cols = ['Age', 'Annual Income']
    for col in numeric_cols:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            
    # For categorical features, fill with mode
    categorical_cols = ['Gender', 'Income Type', 'Employment Status', 'Education', 'Credit History']
    for col in categorical_cols:
        if col in df.columns:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            
    # 3. Remove duplicates
    initial_len = len(df)
    df = df.drop_duplicates()
    print(f"Removed {initial_len - len(df)} duplicate rows.")
    
    # 4. Encode categorical variables using LabelEncoder
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        # Fit on non-null values (already filled)
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        print(f"Encoded column '{col}' using LabelEncoder. Classes: {list(le.classes_)}")
        
    # Save encoders
    os.makedirs('model', exist_ok=True)
    joblib.dump(encoders, 'model/encoders.pkl')
    print("Saved LabelEncoders dictionary to model/encoders.pkl")
    
    # 5. Split features and target
    X = df.drop('Approved', axis=1)
    y = df['Approved']
    
    # 6. Perform train_test_split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train/Test split complete. Training set size: {len(X_train)}, Test set size: {len(X_test)}")
    
    # 7. Scale numerical features using StandardScaler
    scaler = StandardScaler()
    
    # Fit scaler on numerical columns of training set
    X_train_numeric = X_train[numeric_cols]
    X_test_numeric = X_test[numeric_cols]
    
    scaler.fit(X_train_numeric)
    
    # Save scaler
    joblib.dump(scaler, 'model/scaler.pkl')
    print("Saved StandardScaler to model/scaler.pkl")
    
    # Transform numerical features
    X_train_scaled_numeric = scaler.transform(X_train_numeric)
    X_test_scaled_numeric = scaler.transform(X_test_numeric)
    
    # Reassemble X_train and X_test with scaled numerical columns
    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy()
    
    X_train_processed[numeric_cols] = X_train_scaled_numeric
    X_test_processed[numeric_cols] = X_test_scaled_numeric
    
    # 8. Save preprocessed datasets for train_model.py
    os.makedirs('dataset', exist_ok=True)
    X_train_processed.to_csv('dataset/X_train_processed.csv', index=False)
    X_test_processed.to_csv('dataset/X_test_processed.csv', index=False)
    y_train.to_csv('dataset/y_train.csv', index=False)
    y_test.to_csv('dataset/y_test.csv', index=False)
    print("Saved preprocessed training and testing arrays to files.")
    
    return X_train_processed, X_test_processed, y_train, y_test

if __name__ == '__main__':
    preprocess_pipeline()
