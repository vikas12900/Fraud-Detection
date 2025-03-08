import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
xgmodel = pickle.load(open(r"I:\coding\EndToEndML\Fraud-Detection\xgmodel.pkl", 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.json  # Get JSON data
        print("Received JSON:", data)  # Debugging

        # Ensure correct JSON format
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        
        # Ensure input is a flat list
        if not isinstance(data, list):
            return jsonify({'error': 'Input should be a list of values'}), 400

        # Convert to NumPy array and reshape
        input_array = np.array(data, dtype=np.float32).reshape(1, -1)
        print("Processed input shape:", input_array.shape)  # Debugging

        # Validate feature count
        if input_array.shape[1] != 17:
            return jsonify({'error': f'Feature shape mismatch, expected: 17, got {input_array.shape[1]}'}), 400

        # Predict using the model
        output = xgmodel.predict(input_array)

        return jsonify({'prediction': int(output[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)