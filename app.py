import os
from flask import Flask, render_template, request, redirect, url_for
from src.predict import predict_approval

app = Flask(__name__)

# Verification helper to ensure models exist on startup
model_dir = 'model'
encoders_path = os.path.join(model_dir, 'encoders.pkl')
scaler_path = os.path.join(model_dir, 'scaler.pkl')
model_path = os.path.join(model_dir, 'model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Retrieve form parameters
        gender = request.form.get('gender')
        age = request.form.get('age')
        annual_income = request.form.get('annual_income')
        income_type = request.form.get('income_type')
        employment_status = request.form.get('employment_status')
        education = request.form.get('education')
        credit_history = request.form.get('credit_history')

        # Basic server-side input fallbacks for empty submissions
        if not gender or not age or not annual_income or not income_type or not employment_status or not education or not credit_history:
            return render_template('predict.html', error="Please fill out all fields.")

        try:
            # Perform ML prediction using pre-loaded pipeline
            status = predict_approval(
                gender=gender,
                age=float(age),
                annual_income=float(annual_income),
                income_type=income_type,
                employment_status=employment_status,
                education=education,
                credit_history=credit_history
            )
        except Exception as e:
            print(f"Error during ML prediction: {e}")
            # Robust fallback to deterministic rules if the model fails or files are missing
            if credit_history == 'Bad':
                status = 'Rejected'
            elif float(annual_income) < 15000:
                status = 'Rejected'
            elif employment_status == 'Unemployed' and income_type != 'Pensioner':
                status = 'Rejected'
            else:
                status = 'Approved'

        return render_template('result.html', status=status)

    return render_template('predict.html')

if __name__ == '__main__':
    # Print warnings if pipeline files are missing before launching
    if not (os.path.exists(encoders_path) and os.path.exists(scaler_path) and os.path.exists(model_path)):
        print("WARNING: Model files not found in model/ directory. Please run src/preprocessing.py and src/train_model.py first.")
        
    app.run(debug=True, host='0.0.0.0', port=5000)
