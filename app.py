import pickle
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)

# Load trained model
xgmodel = pickle.load(open("xgmodel.pkl", "rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract form data
        data = [float(request.form[key]) for key in request.form.keys()]
        
        # Convert to NumPy array
        final_input = np.array(data).reshape(1, -1)

        # Make predictions
        output = xgmodel.predict(final_input)[0]
        probability = xgmodel.predict_proba(final_input)[0][1]

        return render_template("home.html", prediction_text=f"Prediction: {round(probability, 4)}")

    except Exception as e:
        return render_template("home.html", prediction_text=f"Error: {str(e)}")

# **Handles API Calls (Works with Postman)**
@app.route("/predict_api", methods=["POST"])
def predict_api():
    try:
        # Try to get JSON data
        if request.is_json:
            data = request.get_json()
            if not data or "data" not in data:
                return jsonify({"error": "Invalid or missing JSON data"}), 400
            feature_values = data["data"]
        # Try to get form data
        else:
            feature_values = [float(request.form[key]) for key in request.form.keys()]
       
        # Validate data
        if not isinstance(feature_values, list) or len(feature_values) != 17:
            return jsonify({"error": "Feature shape mismatch, expected: 17 values"}), 400
           
        # Convert to NumPy array
        final_input = np.array(feature_values, dtype=np.float32).reshape(1, -1)
       
        # Make predictions - only get the class prediction (0 or 1)
        prediction = int(xgmodel.predict(final_input)[0])
       
        # Return only the binary prediction (0 or 1)
        return jsonify({
            "prediction": prediction
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == "__main__":
    app.run(debug=True)
