import argparse
import os
import time

import cv2
import imutils
import numpy as np
from imutils.video import FPS, VideoStream

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PROTOTXT = os.path.join(SCRIPT_DIR, "MobileNetSSD_deploy.prototxt.txt")
DEFAULT_MODEL = os.path.join(SCRIPT_DIR, "MobileNetSSD_deploy.caffemodel")

# Argument parser
ap = argparse.ArgumentParser(description="Real-time object detection with MobileNet SSD")
ap.add_argument("-p", "--prototxt", default=DEFAULT_PROTOTXT,
                help="Path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", default=DEFAULT_MODEL,
                help="Path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
                help="Minimum probability to filter weak predictions")
args = vars(ap.parse_args())

if not os.path.isfile(args["prototxt"]):
    raise FileNotFoundError(f"Prototxt file not found: {args['prototxt']}")
if not os.path.isfile(args["model"]):
    raise FileNotFoundError(f"Model file not found: {args['model']}")

# Initialize the class labels for MobileNet SSD
CLASSES = [
    "aeroplane", "background", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "train", "tvmonitor", "paper"
]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("[INFO] Loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

print("[INFO] Starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    frame = vs.read()
    if frame is None:
        break

    frame = imutils.resize(frame, width=800)
    frame = cv2.flip(frame, 1)
    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0 / 127.5,
                                 size=(300, 300), mean=127.5, swapRB=True)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence > args["confidence"]:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            label = f"{CLASSES[idx]}: {confidence * 100:.2f}%"
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, COLORS[idx], 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), ord("Q")):
        break

    fps.update()

fps.stop()
print(f"[INFO] Elapsed Time: {fps.elapsed():.2f}")
print(f"[INFO] Approximate FPS: {fps.fps():.2f}")

cv2.destroyAllWindows()
vs.stop()

