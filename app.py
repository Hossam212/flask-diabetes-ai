from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd

# Load the model
with open('diabetes_model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes and origins

@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['OPTIONS'])
def predict_options():
    return '', 200
def predict():
    data = request.json
    print("Received data:", data)  # Log received data for debugging

    if not data:
        return jsonify({'error': 'No input data provided'}), 400

    try:
        df = pd.DataFrame([data])
        print("DataFrame:", df)  # Log DataFrame for debugging
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
    try:
        prediction = model.predict(df)
        print("Prediction:", prediction)  # Log prediction for debugging
        result = 'Positive' if prediction[0] == 1 else 'Negative'
        return jsonify({'prediction': result}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
