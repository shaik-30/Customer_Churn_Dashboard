import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("data/dataset.csv")

# Remove customerID
if "customerID" in data.columns:
    data.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric
if "TotalCharges" in data.columns:
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

# Remove missing values
data.dropna(inplace=True)

# Separate features and target
X = data.drop("Churn", axis=1)
y = data["Churn"]

# Convert all text columns to numeric
X = pd.get_dummies(X)

# Convert target to numeric
y = y.map({"No": 0, "Yes": 1})

# Check data
print(X.dtypes)
print(X.head())

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/churn_model.pkl")

print("===================================")
print("Model trained successfully!")
print("Model saved as model/churn_model.pkl")
print("===================================")