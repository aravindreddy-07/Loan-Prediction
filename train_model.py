import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv("loan_data.csv")

# Strip whitespace from column names
data.columns = data.columns.str.strip()

# Strip whitespace from 'loan_status' values
data['loan_status'] = data['loan_status'].str.strip()

# Map 'loan_status' to numeric values: Approved -> 1, Rejected -> 0
data['loan_status'] = data['loan_status'].map({'Approved': 1, 'Rejected': 0})

# Drop rows where loan_status is NaN
data = data.dropna(subset=['loan_status'])

# Select features and target variable
X = data[['income_annum', 'loan_amount', 'cibil_score', 
           'residential_assets_value', 'commercial_assets_value', 
           'luxury_assets_value', 'bank_asset_value']]
y = data['loan_status']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

# Perform Cross-Validation and Train each model
best_model = None
best_score = 0
accuracies = {}

for name, model in models.items():
    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    mean_cv_score = cv_scores.mean()
    accuracies[name] = mean_cv_score * 100
    
    print(f"{name} Cross-Validation Accuracy: {mean_cv_score * 100:.2f}%")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Test the model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"{name} Test Accuracy: {accuracy * 100:.2f}%\n")
    
    # Save the best model based on test accuracy
    if accuracy > best_score:
        best_score = accuracy
        best_model = model

# Print final comparison
print("\nModel Comparison:")
for model_name, acc in accuracies.items():
    print(f"{model_name}: {acc:.2f}%")

print(f"\nBest Model: {best_model} with accuracy of {best_score * 100:.2f}%")

# Save the best model
joblib.dump(best_model, "best_loan_model.pkl")
