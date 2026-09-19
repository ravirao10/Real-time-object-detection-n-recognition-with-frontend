import os
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROTOTXT_PATH = os.path.join(SCRIPT_DIR, "MobileNetSSD_deploy.prototxt.txt")
MODEL_PATH = os.path.join(SCRIPT_DIR, "MobileNetSSD_deploy.caffemodel")

CLASSES = [
    "aeroplane", "background", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "train", "tvmonitor", "paper"
]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

app = Flask(__name__, static_folder=None)

camera = None
net = None
current_frame = None
frame_lock = threading.Lock()
is_running = False
stop_event = threading.Event()
worker_thread = None


def load_model():
    global net
    if net is None:
        if not os.path.isfile(PROTOTXT_PATH) or not os.path.isfile(MODEL_PATH):
            raise FileNotFoundError("Required model files are missing.")
        net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)


def detect_and_draw(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0 / 127.5,
                                 size=(300, 300), mean=127.5, swapRB=True)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < 0.2:
            continue

        idx = int(detections[0, 0, i, 1])
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        label = f"{CLASSES[idx]}: {confidence * 100:.2f}%"
        color = COLORS[idx] if COLORS is not None else (0, 255, 0)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        y = startY - 10 if startY - 10 > 10 else startY + 20
        cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2)
    return frame


def camera_loop():
    global camera, current_frame
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera = None
        return

    while not stop_event.is_set():
        ret, frame = camera.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]
        if width > 800:
            scale = 800.0 / width
            frame = cv2.resize(frame, (800, int(height * scale)))

        if net is not None:
            frame = detect_and_draw(frame)

        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        with frame_lock:
            current_frame = jpeg.tobytes()

        time.sleep(0.03)

    if camera is not None:
        camera.release()
        camera = None


@app.route('/')
def index():
    return send_from_directory(SCRIPT_DIR, 'index.html')


@app.route('/start', methods=['POST'])
def start():
    global is_running, worker_thread, stop_event
    if is_running:
        return jsonify(status='running')

    load_model()
    stop_event.clear()
    worker_thread = threading.Thread(target=camera_loop, daemon=True)
    worker_thread.start()
    is_running = True
    return jsonify(status='started')


@app.route('/stop', methods=['POST'])
def stop():
    global is_running
    if not is_running:
        return jsonify(status='stopped')

    stop_event.set()
    if worker_thread is not None:
        worker_thread.join(timeout=2)
    is_running = False
    return jsonify(status='stopped')


@app.route('/status')
def status():
    return jsonify(status='running' if is_running else 'stopped')


@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            if not is_running:
                time.sleep(0.1)
                continue

            with frame_lock:
                frame_data = current_frame

            if frame_data is None:
                time.sleep(0.01)
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
