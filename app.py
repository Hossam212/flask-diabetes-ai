from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd

# Load the model
with open('diabetes_model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    print("Received data:", data)

    if not data:
        return jsonify({'error': 'No input data provided'}), 400

    # Convert the data into a DataFrame
    try:
        df = pd.DataFrame([data])
        print("DataFrame:", df)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    # Replicate preprocessing
    try:
        # Handle categorical encoding
        for col in df.select_dtypes(['object']).columns:
            df[col] = df[col].astype('category')
            df[col] = df[col].cat.codes
        
        print("Processed DataFrame:", df)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    # Make predictions
    try:
        prediction = model.predict(df)
        print("Prediction:", prediction)
        result = 'Positive' if prediction[0] == 1 else 'Negative'
        return jsonify({'prediction': result}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

