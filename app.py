from flask import Flask, request, jsonify
import pickle
import pandas as pd

# Load the model
with open('diabetes_model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
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
