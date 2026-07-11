import os
import joblib
import pandas as pd
import numpy as np

def predict_approval(gender, age, annual_income, income_type, employment_status, education, credit_history):
    """
    Predicts credit card approval based on input features using the trained model.
    """
    model_path = os.path.join('model', 'model.pkl')
    scaler_path = os.path.join('model', 'scaler.pkl')
    encoders_path = os.path.join('model', 'encoders.pkl')
    
    # 1. Verify that all artifacts exist
    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(encoders_path)):
        raise FileNotFoundError("Model artifacts not found. Please train the model first.")
        
    # 2. Load artifacts
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    encoders = joblib.load(encoders_path)
    
    # 3. Create DataFrame of input with generic object type to avoid pandas 3.0 string dtype restrictions
    input_df = pd.DataFrame([{
        'Gender': gender,
        'Age': float(age),
        'Annual Income': float(annual_income),
        'Income Type': income_type,
        'Employment Status': employment_status,
        'Education': education,
        'Credit History': credit_history
    }]).astype(object)
    
    # 4. Label Encode categorical features
    categorical_cols = ['Gender', 'Income Type', 'Employment Status', 'Education', 'Credit History']
    for col in categorical_cols:
        encoder = encoders[col]
        val = input_df.loc[0, col]
        
        # Handle unseen values gracefully
        if val in encoder.classes_:
            encoded_val = encoder.transform([val])[0]
        else:
            encoded_val = encoder.transform([encoder.classes_[0]])[0]
            
        input_df.loc[0, col] = int(encoded_val)
            
    # 5. Scale numerical features
    numeric_cols = ['Age', 'Annual Income']
    scaled_numeric = scaler.transform(input_df[numeric_cols])
    input_df.loc[0, 'Age'] = float(scaled_numeric[0][0])
    input_df.loc[0, 'Annual Income'] = float(scaled_numeric[0][1])
    
    # Convert all columns to float/numeric before feeding to model to prevent dtype warnings
    input_df = input_df.astype(float)
    
    # 6. Predict using the trained model
    prediction = model.predict(input_df)[0]
    
    return 'Approved' if prediction == 1 else 'Rejected'

if __name__ == '__main__':
    # Simple CLI test
    try:
        res = predict_approval(
            gender='Female', 
            age=34, 
            annual_income=85000, 
            income_type='Commercial Associate', 
            employment_status='Employed', 
            education='Higher Education', 
            credit_history='Good'
        )
        print(f"Test prediction output: {res}")
    except Exception as e:
        print(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()
