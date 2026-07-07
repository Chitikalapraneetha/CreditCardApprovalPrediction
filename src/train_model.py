import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

def main():
    print("Loading preprocessed datasets...")
    
    # Check if preprocessed data exists
    if not (os.path.exists('dataset/X_train_processed.csv') and 
            os.path.exists('dataset/X_test_processed.csv') and
            os.path.exists('dataset/y_train.csv') and
            os.path.exists('dataset/y_test.csv')):
        raise FileNotFoundError("Preprocessed data files not found. Please run src/preprocessing.py first.")
        
    X_train = pd.read_csv('dataset/X_train_processed.csv')
    X_test = pd.read_csv('dataset/X_test_processed.csv')
    y_train = pd.read_csv('dataset/y_train.csv').values.ravel()
    y_test = pd.read_csv('dataset/y_test.csv').values.ravel()
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=6),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=8),
        'XGBoost': XGBClassifier(random_state=42, n_estimators=100, max_depth=6, eval_metric='logloss')
    }
    
    trained_models = {}
    model_accuracies = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Predict on test data
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)
        
        # Save results
        trained_models[name] = model
        model_accuracies[name] = acc
        
        # Print metrics
        print(f"--- {name} Performance ---")
        print(f"Accuracy: {acc:.4f}")
        print("Confusion Matrix:")
        print(cm)
        print("Classification Report:")
        print(cr)
        
    # Compare and select best model
    best_model_name = max(model_accuracies, key=model_accuracies.get)
    best_accuracy = model_accuracies[best_model_name]
    best_model = trained_models[best_model_name]
    
    print("\n=========================================")
    print("Model Evaluation Summary:")
    for name, acc in model_accuracies.items():
        print(f" - {name}: {acc * 100:.2f}% Accuracy")
    print(f"\nBest Performing Model: {best_model_name} ({best_accuracy * 100:.2f}% Accuracy)")
    print("=========================================")
    
    # Save best model to model/model.pkl
    os.makedirs('model', exist_ok=True)
    joblib.dump(best_model, 'model/model.pkl')
    print(f"Saved the best model ({best_model_name}) to model/model.pkl")

if __name__ == '__main__':
    main()
