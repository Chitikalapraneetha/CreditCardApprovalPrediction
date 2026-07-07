import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib

def generate_synthetic_data(num_samples=2000):
    np.random.seed(42)
    
    # Generate random features
    gender = np.random.choice(['Male', 'Female'], size=num_samples)
    age = np.random.randint(18, 75, size=num_samples)
    annual_income = np.random.randint(15000, 180000, size=num_samples)
    income_type = np.random.choice(['Working', 'Commercial Associate', 'State Servant', 'Pensioner'], size=num_samples)
    employment_status = np.random.choice(['Employed', 'Unemployed'], size=num_samples)
    education = np.random.choice([
        'Higher Education', 'Secondary / Secondary Special', 
        'Incomplete Higher', 'Lower Secondary', 'Academic Degree'
    ], size=num_samples)
    credit_history = np.random.choice(['Good', 'Bad'], size=num_samples, p=[0.75, 0.25])
    
    # Make labels based on logical scoring with noise
    score = np.zeros(num_samples)
    
    # Positive weights
    score += (credit_history == 'Good') * 0.5
    score += (employment_status == 'Employed') * 0.2
    score += (annual_income / 180000.0) * 0.2
    score += (education == 'Higher Education') * 0.05
    score += (education == 'Academic Degree') * 0.05
    
    # Age penalty / benefit (extreme youth or extreme age without pension can lower score)
    score += ((75 - age) / 57.0) * 0.05
    
    # Unemployed pensioner vs unemployed working age
    for i in range(num_samples):
        if employment_status[i] == 'Unemployed' and income_type[i] != 'Pensioner':
            score[i] -= 0.15
            
    # Add minor noise
    score += np.random.normal(0, 0.05, num_samples)
    
    # Target label: Approved if score >= 0.45
    approved = (score >= 0.45).astype(int)
    
    df = pd.DataFrame({
        'Gender': gender,
        'Age': age,
        'Annual_Income': annual_income,
        'Income_Type': income_type,
        'Employment_Status': employment_status,
        'Education': education,
        'Credit_History': credit_history,
        'Approved': approved
    })
    
    return df

def main():
    print("Generating synthetic credit dataset...")
    df = generate_synthetic_data(3000)
    
    # Save dataset
    os.makedirs('dataset', exist_ok=True)
    df.to_csv('dataset/credit_data.csv', index=False)
    print("Dataset saved to dataset/credit_data.csv")
    
    # Separate features and target
    X = df.drop('Approved', axis=1)
    y = df['Approved']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Preprocessing pipelines
    numeric_features = ['Age', 'Annual_Income']
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_features = ['Gender', 'Income_Type', 'Employment_Status', 'Education', 'Credit_History']
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Complete classification pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42))
    ])
    
    print("Training Random Forest Classifier model...")
    model_pipeline.fit(X_train, y_train)
    
    # Evaluate model
    train_acc = model_pipeline.score(X_train, y_train)
    test_acc = model_pipeline.score(X_test, y_test)
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Testing Accuracy: {test_acc:.4f}")
    
    # Save the pipeline
    os.makedirs('model', exist_ok=True)
    joblib.dump(model_pipeline, 'model/model.pkl')
    print("Model pipeline successfully saved to model/model.pkl")

if __name__ == '__main__':
    main()
