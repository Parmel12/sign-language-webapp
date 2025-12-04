import os
import io
import base64
import threading
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Mediapipe & ML
import mediapipe as mp
import joblib

# ------------------ Configuration ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "sign_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")

ALLOWED_SIGNS = {"Hello", "Thank you", "Yes", "Ok", "Please",
                 "Sorry", "Help", "I love you", "peace", "play"}

# ------------------ Validate model files ------------------
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(
        "Make sure 'sign_model.pkl' and 'scaler.pkl' are in the 'model/' folder."
    )

# ------------------ Load model & scaler ------------------
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ------------------ Mediapipe setup ------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# Create hands object once (thread-safe caution: we'll use a lock around .process)
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
hands_lock = threading.Lock()

# ------------------ Flask app ------------------
app = Flask(__name__, static_folder=".", static_url_path="/")
CORS(app)  # allow cross-origin from frontend

def extract_landmarks_from_results(results):
    """Return features list or None."""
    if not results.multi_hand_landmarks:
        return None
    lm = results.multi_hand_landmarks[0].landmark
    base_x, base_y = lm[0].x, lm[0].y
    features = []
    for p in lm:
        features.extend([p.x - base_x, p.y - base_y])
    return features

def pil_from_base64(base64_str):
    """Decode base64 image (dataURL or raw base64) and return PIL Image (RGB)."""
    if base64_str.startswith("data:"):
        base64_str = base64_str.split(",", 1)[1]
    img_bytes = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")

@app.route("/")
def index():
    # Serve the index.html in the project root (so Render can serve it too if needed)
    return send_from_directory(".", "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON: { "image": "<base64 dataurl or base64 string>" }
    Returns JSON: { "sign": "Hello", "confidence": 92.3 } or { "sign": null, "confidence": 0 }
    """
    data = request.get_json(force=True)
    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    try:
        pil_img = pil_from_base64(data["image"])
    except Exception as e:
        return jsonify({"error": "Invalid image data", "details": str(e)}), 400

    # Resize to reasonable size for performance
    pil_img.thumbnail((640, 480))
    np_img = np.array(pil_img)  # RGB

    # Mediapipe expects RGB
    with hands_lock:
        results = hands.process(np_img)

    features = extract_landmarks_from_results(results)
    if features is None:
        return jsonify({"sign": None, "confidence": 0.0})

    # Scale & predict
    try:
        X = scaler.transform([features])
    except Exception as e:
        return jsonify({"error": "Scaler transform failed", "details": str(e)}), 500

    try:
        pred = model.predict(X)[0]
        prob = 0.0
        try:
            prob = float(np.max(model.predict_proba(X)) * 100)
        except Exception:
            # model may not support predict_proba
            prob = None
    except Exception as e:
        return jsonify({"error": "Model prediction failed", "details": str(e)}), 500

    if prob is not None:
        prob = round(float(prob), 2)

    if prob is not None and prob < 70:
        # low confidence -> return None
        return jsonify({"sign": None, "confidence": prob})

    if pred not in ALLOWED_SIGNS:
        # Not in allowed set
        return jsonify({"sign": None, "confidence": prob if prob is not None else 0})

    return jsonify({"sign": str(pred), "confidence": prob if prob is not None else 100.0})

if __name__ == "__main__":
    # For local testing use this; Render will use gunicorn as specified in Procfile
    app.run(host="0.0.0.0", port=5000, debug=False)
