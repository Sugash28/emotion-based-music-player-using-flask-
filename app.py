from flask import Flask, render_template, Response, jsonify
import cv2
from flask_cors import CORS
import numpy as np
import mediapipe as mp
from keras.models import load_model
import time

app = Flask(__name__)
CORS(app)

model = load_model("model.h5")
label = np.load("labels.npy")

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    holistic = mp.solutions.holistic
    hands = mp.solutions.hands
    holis = holistic.Holistic()
    drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    start_time = time.time()
    emotion_detected = None

    while True:
        lst = []

        _, frm = cap.read()

        frm = cv2.flip(frm, 1)

        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                for _ in range(42):
                    lst.append(0.0)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                for _ in range(42):
                    lst.append(0.0)

            lst = np.array(lst).reshape(1, -1)

            emotion_detected = label[np.argmax(model.predict(lst))]
            emotion_text = f"Emotion: {emotion_detected}"
            cv2.putText(frm, emotion_text, (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)

            if time.time() - start_time >= 10:
                break

        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        ret, buffer = cv2.imencode('.jpg', frm)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

    final_emotion_text = f"Final Emotion Detected: {emotion_detected}"
    final_frame = np.ones((100, 600, 3), np.uint8) * 255  
    cv2.putText(final_frame, final_emotion_text, (50, 50), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)
    ret, buffer = cv2.imencode('.jpg', final_frame)
    final_frame_bytes = buffer.tobytes()
    yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + final_frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/emotion', methods=['GET'])
def get_emotion():
    emotion = detect_emotion()
    return jsonify({'emotion': emotion})

def detect_emotion():
    holistic = mp.solutions.holistic
    hands = mp.solutions.hands
    holis = holistic.Holistic()
    drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    start_time = time.time()
    emotion_detected = None

    while True:
        lst = []

        _, frm = cap.read()

        frm = cv2.flip(frm, 1)

        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                for _ in range(42):
                    lst.append(0.0)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                for _ in range(42):
                    lst.append(0.0)

            lst = np.array(lst).reshape(1, -1)

            emotion_detected = label[np.argmax(model.predict(lst))]
            if time.time() - start_time >= 10:
                break

    cap.release()
    cv2.destroyAllWindows()

    return emotion_detected

if __name__ == "__main__":
    app.run(debug=True)
