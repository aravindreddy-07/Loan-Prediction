from flask import Flask, render_template, request
import joblib

# Load your model
model = joblib.load('loan_model.pkl')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    income_annum = float(request.form['income_annum'])
    loan_amount = float(request.form['loan_amount'])
    cibil_score = float(request.form['cibil_score'])
    residential_assets_value = float(request.form['residential_assets_value'])
    commercial_assets_value = float(request.form['commercial_assets_value'])
    luxury_assets_value = float(request.form['luxury_assets_value'])
    bank_asset_value = float(request.form['bank_asset_value'])
    
    # Make prediction
    prediction = model.predict([[income_annum, loan_amount, cibil_score, 
                                 residential_assets_value, commercial_assets_value, 
                                 luxury_assets_value, bank_asset_value]])
    
    # Convert prediction to readable text
    prediction_text = 'Approved' if prediction[0] == 1 else 'Rejected'
    
    # Render result page with prediction
    return render_template('result.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
