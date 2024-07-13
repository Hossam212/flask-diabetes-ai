from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import json
from keras.src.legacy.saving import legacy_h5_format


app = Flask(__name__)
model = legacy_h5_format.load_model_from_hdf5('RainfallReveal.h5', custom_objects={'mae': 'mae'})

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
