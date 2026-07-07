# Credit Card Approval Prediction System

A modern, professional, and responsive web application UI built for machine learning-based credit card approval predictions. This template is fully configured as a Python Flask application structure utilizing Bootstrap 5, custom styles, and clean dynamic rendering templates.

## Tech Stack
- **Backend:** Python, Flask
- **Frontend:** HTML5 (Jinja2), CSS3 (Custom design system), JavaScript
- **Frameworks:** Bootstrap 5, Bootstrap Icons
- **ML Core (Placeholders for integration):** Scikit-Learn, Pandas, NumPy, XGBoost

---

## Directory Structure
```text
CreditCardApprovalPrediction/
│
├── app.py                  # Main Flask backend application server
├── requirements.txt        # Python package dependencies
├── README.md               # Project documentation
│
├── dataset/                # Placeholder folder for CSV dataset training inputs
├── model/                  # Placeholder folder for compiled pickle models (.pkl)
├── src/                    # Placeholder folder for processing and training pipelines
│
├── templates/              # HTML layout templates (Jinja2)
│   ├── index.html          # Hero page & system benefits overview
│   ├── about.html          # Explains algorithms (Logistic Regression, Random Forest, etc.)
│   ├── predict.html        # Clean, responsive user evaluation form
│   └── result.html         # Renders dynamic Green Approved / Red Rejected results
│
└── static/                 # Static assets folder
    ├── css/
    │   └── style.css       # Core branding stylesheet (fintech theme variables)
    ├── js/
    │   └── script.js       # Client validations & form triggers
    └── images/
        └── hero.png        # Premium 3D fintech credit card illustration
```

---

## Quick Start Guide

### 1. Prerequisites
Ensure you have Python 3.8+ installed on your workspace.

### 2. Environment Setup
Install the necessary requirements listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 3. Running the Server
Execute the main application file to spin up a local development server:
```bash
python app.py
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your web browser to interact with the application dashboard.

---

## Prediction Parameters
The model handles 12 diagnostic parameters to calculate loan risk probability thresholds:
1. **Gender:** Male or Female
2. **Age:** Input value validated between 18 and 120
3. **Annual Income:** Annual pre-tax earnings ($)
4. **Income Type:** Working, Commercial Associate, State Servant, Pensioner
5. **Employment Status:** Employed or Unemployed
6. **Years Employed:** Total tenure in years
7. **Education Level:** Higher Education, Secondary, etc.
8. **Marital Status:** Married, Single, Separated, etc.
9. **Housing Type:** Ownership profiles (House, Rented, etc.)
10. **Credit History:** Past records of defaults or updates
11. **Existing Loan:** Indicates outstanding financial commitments
12. **Family Members:** Dependents calculation (min 1)

---

## Modifying ML Inference
Currently, `app.py` runs a logical deterministic ruleset to mimic prediction classification.
To link your trained model (e.g., `model/xgboost_model.pkl`):
1. Import `pickle` or `joblib` inside `app.py`.
2. Load the model during application startup:
   ```python
   model = pickle.load(open('model/xgboost_model.pkl', 'rb'))
   ```
3. Update the `/predict` POST handler to convert form parameters into a numpy array / pandas DataFrame and call `model.predict()`.
