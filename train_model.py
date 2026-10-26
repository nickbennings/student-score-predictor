import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

# Load the dataset
data = pd.read_csv("student_scores.csv")

# Select the input and the value we want to predict
X = data[["hours_studied"]]
y = data["exam_score"]

# Set aside some data to test the model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Test the model on data it hasn't seen during training
predictions = model.predict(X_test)
error = mean_absolute_error(y_test, predictions)

print(f"Model trained! Average prediction error: {error:.2f} points")

# Save the trained model for the web app
joblib.dump(model, "student_score_model.pkl")
print("Model saved as student_score_model.pkl")

