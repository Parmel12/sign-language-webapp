from flask import Flask, Response, render_template_string
import cv2
import mediapipe as mp
import joblib
import numpy as np
import pyttsx3
import threading
import queue
import sys

# ================= LOAD MODEL =================
model = joblib.load("sign_model.pkl")
scaler = joblib.load("scaler.pkl")

ALLOWED_SIGNS = {
    "Hello", "Thank you", "Yes", "Ok", "Please",
    "Sorry", "Help", "I love you", "peace", "play"
}

# ================= VOICE ENGINE =================
engine = pyttsx3.init()
engine.setProperty('rate', 180)
engine.setProperty('volume', 1.0)

speech_queue = queue.Queue()

def speech_worker():
    while True:
        text = speech_queue.get()
        if text is None:
            break
        try:
            engine.stop()
            engine.say(text)
            engine.runAndWait()
        except:
            pass

threading.Thread(target=speech_worker, daemon=True).start()

# ================= CAMERA SETUP =================
def open_camera():
    for cam_id in [0, 1, 2]:
        cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if cap.isOpened():
            print(f"Camera opened successfully (ID={cam_id})")
            return cap
        cap.release()
    print("ERROR: No camera found.")
    sys.exit(1)

cap = open_camera()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

last_spoken_sign = None

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Sign Language AI</title>
</head>
<body style="background:black; text-align:center;">
<h1 style="color:white;">Sign Language AI Stream</h1>
<img src="/video_feed" width="100%">
</body>
</html>
"""

def extract_landmarks(results):
    if not results.multi_hand_landmarks:
        return None
    lm = results.multi_hand_landmarks[0].landmark
    base_x, base_y = lm[0].x, lm[0].y
    return [(p.x-base_x, p.y-base_y) for p in lm]

def gen_frames():
    global last_spoken_sign
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        sign_text = "No Hand"
        current_sign = None

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            landmarks = extract_landmarks(results)
            if landmarks:
                X = scaler.transform([np.array(landmarks).flatten()])
                prediction = model.predict(X)[0]
                confidence = np.max(model.predict_proba(X)) * 100

                if confidence > 70 and prediction in ALLOWED_SIGNS:
                    current_sign = prediction
                    sign_text = f"{prediction} ({confidence:.0f}%)"

                    if last_spoken_sign != current_sign:
                        while not speech_queue.empty():
                            speech_queue.get_nowait()
                        speech_queue.put(current_sign)
                        last_spoken_sign = current_sign

        else:
            last_spoken_sign = None

        cv2.putText(frame, sign_text, (10, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("📱 OPEN ON PHONE: http://YOUR_PC_IP:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
