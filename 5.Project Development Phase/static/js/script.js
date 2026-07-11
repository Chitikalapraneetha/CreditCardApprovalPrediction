/**
 * Credit Card Approval Prediction System - JavaScript Controls
 */

document.addEventListener('DOMContentLoaded', () => {
    // Form Validation and Interactive Indicators
    const predictForm = document.getElementById('predictForm');
    
    if (predictForm) {
        predictForm.addEventListener('submit', (event) => {
            if (!predictForm.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            predictForm.classList.add('was-validated');
        }, false);

        // Add Real-time visual aids for numerical values
        const ageInput = document.getElementById('age');
        if (ageInput) {
            ageInput.addEventListener('input', (e) => {
                const val = parseFloat(e.target.value);
                if (val < 18 || val > 120) {
                    ageInput.setCustomValidity('Age must be between 18 and 120');
                } else {
                    ageInput.setCustomValidity('');
                }
            });
        }

        const incomeInput = document.getElementById('annual_income');
        if (incomeInput) {
            incomeInput.addEventListener('input', (e) => {
                const val = parseFloat(e.target.value);
                if (val < 0) {
                    incomeInput.setCustomValidity('Income cannot be negative');
                } else {
                    incomeInput.setCustomValidity('');
                }
            });
        }
    }
});
