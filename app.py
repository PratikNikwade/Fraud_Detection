"""
DocSecure Flask API — Fraud Detection Backend
Loads the trained EfficientNetB0 model and serves predictions.
"""

import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
import keras

# ── Patch: strip unsupported 'quantization_config' from saved configs ──
# The model was saved with a newer Keras that includes quantization_config
# in Dense/Conv2D layers, which the current Keras version doesn't recognize.
_LAYERS_TO_PATCH = [keras.layers.Dense, keras.layers.Conv2D, keras.layers.BatchNormalization]
for _layer_cls in _LAYERS_TO_PATCH:
    _orig_from_config = _layer_cls.from_config.__func__ if hasattr(_layer_cls.from_config, '__func__') else _layer_cls.from_config

    @classmethod
    def _patched_from_config(cls, config, _orig=_orig_from_config):
        config.pop('quantization_config', None)
        return _orig(cls, config)

    _layer_cls.from_config = _patched_from_config

# ── App Setup ───────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from the frontend

# ── Load Model ──────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_fraud_model.keras')

print("[*] Loading model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print(f"[OK] Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    model = None

IMG_SIZE = (224, 224)
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_risk_level(confidence):
    if confidence >= 0.95:
        return "Very High"
    elif confidence >= 0.85:
        return "High"
    elif confidence >= 0.70:
        return "Moderate"
    elif confidence >= 0.50:
        return "Low"
    else:
        return "Very Low"


# ── Prediction Endpoint ────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Place best_fraud_model.keras in the project root.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded. Send an image with the key "file".'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'Empty filename.'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': f'Unsupported file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

    try:
        # Read and preprocess the image
        img = Image.open(file.stream).convert('RGB')
        img = img.resize(IMG_SIZE, Image.LANCZOS)
        img_array = np.array(img, dtype=np.float32)          # pixels in [0, 255]
        img_array = np.expand_dims(img_array, axis=0)        # batch dimension → (1, 224, 224, 3)

        # Run inference
        prediction = model.predict(img_array, verbose=0)
        prob_fake = float(prediction[0][0])                   # sigmoid output: probability of FAKE (class 1)

        # Determine label and confidence
        if prob_fake >= 0.5:
            label = 'FAKE'
            confidence = prob_fake
        else:
            label = 'REAL'
            confidence = 1.0 - prob_fake

        return jsonify({
            'label': label,
            'confidence': round(confidence, 4),
            'probability': round(prob_fake, 6),
            'risk_level': get_risk_level(confidence),
            'filename': file.filename,
        })

    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500


# ── Health Check ────────────────────────────────────────
@app.route('/', methods=['GET'])
def health():
    return jsonify({
        'status': 'running',
        'model_loaded': model is not None,
        'message': 'DocSecure Fraud Detection API'
    })


# ── Run ─────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n[*] DocSecure API running at http://127.0.0.1:5000")
    print("    POST /predict  - upload an image for fraud detection")
    print("    GET  /         - health check\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
