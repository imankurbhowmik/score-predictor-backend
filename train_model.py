import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import os

# Paths
DATA_FILE = os.path.join(os.path.dirname(__file__), "student_scores.csv")
MODEL_FILE = os.path.join(os.path.dirname(__file__), "model.pkl")

# Load CSV
df = pd.read_csv(DATA_FILE)

# Optional: show columns
print("Columns in file:", df.columns.tolist())

# Make sure they match your file
X = df[['Hours']]
y = df['Scores']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
with open(MODEL_FILE, "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as model.pkl")



