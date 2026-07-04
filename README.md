# Loan-Prediction

A machine learning web application for predicting loan approval status using applicant financial information.

The project includes:
- Model training and comparison across multiple ML algorithms
- A Flask web app for interactive predictions
- Pre-trained model files (`.pkl`) for quick deployment
- HTML/CSS frontend pages for user input and result display

---

## Features

- ✅ Predicts loan status: **Approved** or **Rejected**
- ✅ Flask backend for serving predictions
- ✅ Clean web flow with `index`, `form`, and `result` pages
- ✅ Model training pipeline with multiple algorithms:
  - Logistic Regression
  - K-Nearest Neighbors
  - Support Vector Machine
  - Decision Tree
  - Random Forest
  - Gradient Boosting
- ✅ Cross-validation + test accuracy comparison during training
- ✅ Saved model artifacts using `joblib`

---

## Tech Stack

- **Backend:** Python, Flask
- **Machine Learning:** scikit-learn, pandas, NumPy, joblib
- **Frontend:** HTML, CSS

---

## Project Structure

```text
Loan-Prediction/
├── app.py                         # Flask application
├── train_model.py                 # Train and compare ML models
├── test_model.py                  # Simple model save/load test script
├── requirements.txt               # Python dependencies
├── loan_data.csv                  # Training dataset
├── index.html                     # Landing page
├── form.html                      # Input form page
├── result.html                    # Prediction result page
├── style.css                      # Styling
├── favicon.ico                    # Favicon asset
├── loan_model.pkl                 # Saved model (used by app.py)
├── best_loan_model.pkl            # Best model from training script
├── best_loan_model_lightgbm.pkl   # Additional saved model
├── logistic_model.pkl
├── knn_model.pkl
├── svm_model.pkl
├── decision_tree_model.pkl
├── random_forest_model.pkl
├── test_model.pkl
└── README.md
```

---

## Installation

```bash
# 1) Clone the repository
git clone https://github.com/aravindreddy-07/Loan-Prediction.git
cd Loan-Prediction

# 2) (Optional) Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

# 3) Install dependencies
pip install -r requirements.txt
```

---

## Run the Web App

```bash
python app.py
```

Then open your browser and go to:

```text
http://127.0.0.1:5000/
```

---

## How Prediction Works

The Flask app (`app.py`) loads a pre-trained model:

```python
model = joblib.load('loan_model.pkl')
```

It accepts these 7 numerical inputs from the form:
- `income_annum`
- `loan_amount`
- `cibil_score`
- `residential_assets_value`
- `commercial_assets_value`
- `luxury_assets_value`
- `bank_asset_value`

Model output is mapped to:
- `1` → **Approved**
- `0` → **Rejected**

---

## Train Models Again (Optional)

To retrain models and compare performance:

```bash
python train_model.py
```

What this script does:
1. Loads `loan_data.csv`
2. Cleans `loan_status` values
3. Maps labels (`Approved`/`Rejected`) to `1/0`
4. Scales features with `StandardScaler`
5. Trains multiple models
6. Evaluates via cross-validation + test split
7. Saves the best model as `best_loan_model.pkl`

> Note: `app.py` currently uses `loan_model.pkl`. If you want the app to use the newly trained best model, update `app.py` accordingly.

---

## Requirements

From `requirements.txt`:
- Flask==2.0.1
- scikit-learn==1.0
- pandas==1.3.3
- joblib==1.0.1

---

## Known Notes

- `app.py` expects template files (`index.html`, `form.html`, `result.html`) to be discoverable by Flask. If routing/templates fail, move HTML files into a `templates/` directory and CSS into `static/`.
- Multiple model `.pkl` files are present; ensure the one loaded in `app.py` is the intended production model.

---

## Future Improvements

- Add input validation and better error handling in forms
- Add model confidence/probability score in results
- Add API endpoint (`/predict-api`) for programmatic use
- Use a pipeline object (scaler + model) for safer inference
- Add Dockerfile and deployment instructions

---

## License

No license file is currently present in this repository.
If you plan to share or reuse this project publicly, consider adding a license (e.g., MIT).
