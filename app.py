from flask import Flask, request, jsonify
import numpy as np
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

app = Flask(__name__)

MODEL_FILE = "house_price_model.pkl"

def train_model():
    # ✅ Use California housing dataset instead of deprecated Boston dataset
    housing = fetch_california_housing()
    X = housing.data
    y = housing.target

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save model and feature names
    joblib.dump((model, housing.feature_names), MODEL_FILE)
    print("✅ Model trained and saved as house_price_model.pkl")

# Load or train model
if not os.path.exists(MODEL_FILE):
    train_model()
else:
    print("✅ Loaded saved model.")

model, feature_names = joblib.load(MODEL_FILE)

@app.route('/')
def home():
    return "🏡 California House Price Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(features)[0]
        return jsonify({
            "predicted_price": round(float(prediction), 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
