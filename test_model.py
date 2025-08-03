import joblib
from sklearn.linear_model import LogisticRegression
import numpy as np

# Create a simple model
model = LogisticRegression()
# Dummy data for training
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([0, 1, 0])
model.fit(X, y)

# Save the model
joblib.dump(model, 'test_model.pkl')

# Load the model
loaded_model = joblib.load('test_model.pkl')

# Test the model
prediction = loaded_model.predict([[3, 4]])
print("Prediction:", prediction)
