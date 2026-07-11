import os
import numpy as np
import pandas as pd

def generate_synthetic_dataset(filepath, num_samples=3000):
    np.random.seed(42)
    
    # Generate columns
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
    
    # Label generation logic (deterministic with random noise)
    score = np.zeros(num_samples)
    score += (credit_history == 'Good') * 0.5
    score += (employment_status == 'Employed') * 0.2
    score += (annual_income / 180000.0) * 0.2
    score += (education == 'Higher Education') * 0.05
    score += (education == 'Academic Degree') * 0.05
    score += ((75 - age) / 57.0) * 0.05
    
    for i in range(num_samples):
        if employment_status[i] == 'Unemployed' and income_type[i] != 'Pensioner':
            score[i] -= 0.15
            
    score += np.random.normal(0, 0.05, num_samples)
    approved = (score >= 0.45).astype(int)
    
    df = pd.DataFrame({
        'Gender': gender,
        'Age': age,
        'Annual Income': annual_income,
        'Income Type': income_type,
        'Employment Status': employment_status,
        'Education': education,
        'Credit History': credit_history,
        'Approved': approved
    })
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Generated synthetic dataset and saved to {filepath}")

if __name__ == '__main__':
    generate_synthetic_dataset('dataset/credit_card.csv')
