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
    # Convert the data into a DataFrame
    df = pd.DataFrame(data, index=[0])
    
    # Make predictions
    prediction = model.predict(df)
    result = 'Positive' if prediction[0] == 1 else 'Negative'
    
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
