from flask import Flask, request, render_template
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Modeli ve ölçekleyiciyi yükleme
model = load_model('model.h5')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    input_array = np.array([input_features])
    scaled_input = scaler.transform(input_array)
    prediction = model.predict(scaled_input)
    prediction_probs = prediction[0]
    predicted_class = np.argmax(prediction_probs)
    classes = ['low', 'medium', 'high']
    result = classes[predicted_class]
    probabilities = {c: f"{p*100:.2f}%" for c, p in zip(classes, prediction_probs)}
    return render_template('index.html', prediction_text=result, probabilities=probabilities)

if __name__ == "__main__":
    app.run(debug=True)
