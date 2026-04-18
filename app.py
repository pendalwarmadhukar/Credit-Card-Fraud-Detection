from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model and scalers on startup
try:
    model = joblib.load('fraud_model.pkl')
    scaler_amount = joblib.load('scaler_amount.pkl')
    scaler_time = joblib.load('scaler_time.pkl')
    print("Model and Scalers loaded successfully.")
except Exception as e:
    print(f"Error loading model artifacts: {e}")
    model = None
    scaler_amount = None
    scaler_time = None

@app.route('/', methods=['GET'])
def index():
    """Serve the dashboard UI."""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    """Endpoint to check API health."""
    return jsonify({
        "model_name": "Credit Card Fraud detector",
        "version": "1.0.0",
        "status": "UP" if model is not None else "DOWN"
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to make predictions."""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400
        
        # Expected features order (excluding Class)
        # Time, V1-V28, Amount -> After scaling/dropping we have V1-V28, scaled_amount, scaled_time
        
        # Create DataFrame from input
        input_df = pd.DataFrame([data])
        
        # Preprocessing: Scale Time and Amount
        if 'Amount' in input_df.columns:
            input_df['scaled_amount'] = scaler_amount.transform(input_df[['Amount']])
        else:
            return jsonify({"error": "Missing 'Amount' field"}), 400
            
        if 'Time' in input_df.columns:
            input_df['scaled_time'] = scaler_time.transform(input_df[['Time']])
        else:
            return jsonify({"error": "Missing 'Time' field"}), 400

        # Features order: V1-V28, scaled_amount, scaled_time
        feature_cols = [f'V{i}' for i in range(1, 29)] + ['scaled_amount', 'scaled_time']
        
        X_input = input_df[feature_cols]
        
        # Make prediction
        prob = model.predict_proba(X_input)[:, 1][0]
        prediction = int(prob >= 0.3) # Using recommended threshold
        
        # Determine risk level
        risk_level = "LOW"
        if prob > 0.7:
            risk_level = "HIGH"
        elif prob > 0.3:
            risk_level = "MEDIUM"
            
        return jsonify({
            "is_fraud": bool(prediction),
            "fraud_probability": round(float(prob), 4),
            "risk_level": risk_level
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Usage: Sample curl command
    # curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"Time\": 1000, \"Amount\": 150.5, \"V1\": -1.35, ...}"
    app.run(debug=True, port=5000)
