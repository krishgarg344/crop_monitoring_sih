# app.py
from flask import Flask
import tensorflow as tf
import numpy as np

# --- 1. Create the Flask App ---
app = Flask(__name__)

# --- 2. Load the Trained Model ---
# Make sure the model file is in the same directory as this script
try:
    model = tf.keras.models.load_model('model/crop_health_model.keras')
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# --- 3. Define Class Names ---
class_names = ['High Stress / Soil', 'Moderate Stress', 'Healthy']

# --- 4. Define Routes ---
@app.route("/")
def hello_world():
    """Homepage to confirm the server is running."""
    return "<p>Hello, World! Our Python server is running!</p>"

@app.route("/predict")
def predict():
    """Prediction route to test the model."""
    if model is None:
        return "Model not loaded. Please check the server logs."

    # Create a single dummy image patch (1 patch, 64x64 pixels, 3 channels)
    # The values are random, so the prediction will also be random.
    dummy_patch = np.random.rand(1, 64, 64, 3) * 255
    
    # Get the model's prediction
    prediction = model.predict(dummy_patch)
    
    # Get the index of the highest probability
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    
    # Get the corresponding human-readable label
    predicted_label = class_names[predicted_class_index]
    
    return f"<h1>Random Patch Prediction</h1><p>The model predicts: <strong>{predicted_label}</strong></p>"

# This part is optional but good practice
if __name__ == '__main__':
    app.run(debug=True)