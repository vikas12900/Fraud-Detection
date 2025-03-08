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
        data = request.json["data"]
        input_array = np.array(data, dtype=np.float32).reshape(1, -1)
        proba = xgmodel.predict_proba(input_array)[0][1]  # Get fraud probability

        return jsonify({'prediction': round(float(proba), 4)})  # Round to 4 decimals

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == "__main__":
    app.run(debug=True)