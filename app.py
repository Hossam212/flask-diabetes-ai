from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import json

app = Flask(__name__)
model = load_model('RainfallReveal.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    weather_data = data['weather_data']
    
    # Assuming the model expects a 3D array: (samples, time steps, features)
    processed_data = np.array(weather_data).reshape((1, len(weather_data), len(weather_data[0])))
    
    prediction = model.predict(processed_data)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
