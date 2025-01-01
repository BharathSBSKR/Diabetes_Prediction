from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv('diabetes.csv')

# Split features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Preprocess data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model and scaler
with open('diabetes_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Create Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input features
        input_features = [float(x) for x in request.form.values()]
        
        # Load scaler and model
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        with open('diabetes_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Preprocess input
        input_scaled = scaler.transform([input_features])
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Return result
        result = "Diabetes Detected" if prediction == 1 else "No Diabetes Detected"
        return render_template('index.html', prediction=result)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
