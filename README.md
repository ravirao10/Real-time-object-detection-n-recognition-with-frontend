
Real-Time Object Detection & Recognition
<p align="center"> <strong>Detect objects from a live webcam stream using OpenCV, MobileNet SSD, and a Flask-powered web interface.</strong> </p><p align="center"> <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white" alt="Python"> <img src="https://img.shields.io/badge/OpenCV-DNN-green?logo=opencv&logoColor=white" alt="OpenCV"> <img src="https://img.shields.io/badge/Model-MobileNet%20SSD-orange" alt="MobileNet SSD"> <img src="https://img.shields.io/badge/Backend-Flask-black?logo=flask" alt="Flask"> <img src="https://img.shields.io/badge/Interface-HTML%20%7C%20CSS%20%7C%20JavaScript-E34F26?logo=html5&logoColor=white" alt="Frontend"> </p><p align="center"> <a href="https://github.com/ravirao10/Real-time-object-detection-n-recognition-with-frontend">Repository</a> · <a href="#-getting-started">Getting started</a> · <a href="#-project-structure">Project structure</a> </p>
👀 Overview
Real-Time Object Detection & Recognition is a computer vision project that identifies objects from a live webcam feed and displays the detected objects with bounding boxes and confidence scores.

The project uses MobileNet SSD, a lightweight object detection model, with OpenCV's Deep Neural Network (DNN) module for processing video frames. It includes a browser-based interface powered by Flask, as well as a standalone Python application.

The goal is to make real-time object detection accessible through a simple interface while keeping the detection pipeline suitable for local execution.

✨ Features
Live webcam detection — captures frames from the computer's default camera.
Object identification — detects supported object categories using the pretrained MobileNet SSD model.
Bounding-box visualization — highlights detected objects directly on the video frame.
Confidence scores — displays the model's confidence percentage for each detection.
Browser-based controls — start and stop the camera stream from the web interface.
Live video streaming — sends processed frames from Flask to the browser using an MJPEG stream.
Standalone execution — run detection directly from a Python script without the web interface.
Configurable detection threshold — adjust the minimum confidence for the standalone script through a command-line argument.
🖥️ How it works
The application follows this basic pipeline:

Webcam
  │
  ▼
Capture video frames
  │
  ▼
Preprocess frame with OpenCV
  │
  ▼
MobileNet SSD (Caffe model)
  │
  ▼
Object detections + confidence scores
  │
  ▼
Draw bounding boxes and labels
  │
  ├──► Flask + MJPEG stream ──► Browser interface
  │
  └──► OpenCV display window ──► Standalone application
For the web application, the browser sends start/stop requests to Flask. Flask manages the camera-processing thread and exposes the latest processed frame through the video-feed endpoint.

🧰 Technology stack
Technology	Purpose
Python	Core application and detection logic
OpenCV	Camera access, image processing, DNN inference, and drawing
MobileNet SSD	Pretrained object detection model
Caffe	Model and network configuration format
Flask	Web server and video-stream endpoints
NumPy	Array and bounding-box calculations
HTML, CSS, JavaScript	Browser-based user interface
imutils	Video-stream and image-resizing utilities for standalone mode
📦 Project structure
Real-time-object-detection-n-recognition-with-frontend/
│
├── app.py                              # Flask web application
├── index.html                          # Browser interface
├── style.css                           # Additional stylesheet
├── script.ts.txt                       # Standalone event-handling reference
│
├── real_time_object_detection.py       # Standalone webcam detection
├── test_cam - Copy.py                  # Camera test script
│
├── MobileNetSSD_deploy.prototxt.txt    # MobileNet SSD network configuration
├── MobileNetSSD_deploy.caffemodel      # Pretrained model weights
│
├── LICENSE
└── README.md
The standalone detector and Flask web app are separate entry points. Choose the one you want to run. The current index.html contains its own inline CSS and JavaScript; style.css and script.ts.txt are separate files and are not linked by that page.

🚀 Getting started
1. Prerequisites
Make sure you have:

Python 3 installed.
A working webcam connected to your computer.
Git installed.
A browser for the web application.
2. Clone the repository
git clone https://github.com/ravirao10/Real-time-object-detection-n-recognition-with-frontend.git
cd Real-time-object-detection-n-recognition-with-frontend
3. Create a virtual environment (recommended)
Windows:

python -m venv venv
venv\Scripts\activate
macOS / Linux:

python3 -m venv venv
source venv/bin/activate
4. Install dependencies
For the Flask web application:

pip install opencv-python numpy Flask
For the standalone Python application, also install:

pip install imutils
The project currently does not include a requirements.txt; these commands install the dependencies imported by the application. The OpenCV DNN module is included in standard modern OpenCV Python packages.

▶️ Run the project
Option A: Browser-based application (Flask)
Start the Flask server:

python app.py
Open the following address in your browser:

http://127.0.0.1:5000
Then:

Open the application in your browser.
Select Start to begin webcam capture and object detection.
View detected objects, bounding boxes, and confidence scores in the live video.
Select Stop to end camera capture.
The interface communicates with these Flask endpoints:

Endpoint	Method	Purpose
/	GET	Serves the web interface
/start	POST	Loads the model and starts camera processing
/stop	POST	Stops camera processing
/status	GET	Returns the camera-processing status
/video_feed	GET	Streams processed frames as MJPEG
Option B: Standalone Python application
Run the detector:

python real_time_object_detection.py
The script uses the model files in the repository by default. To customize the detection confidence threshold:

python real_time_object_detection.py --confidence 0.4
You can also explicitly provide the model files:

python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel --confidence 0.2
To see the available command-line options:

python real_time_object_detection.py --help
Press Q in the OpenCV display window to exit the standalone application.

🎯 Object categories
The application contains a class-label list in its Python source code, covering common categories such as people, vehicles, animals, furniture, and household objects. The exact label-to-class mapping must match the pretrained Caffe model being used. If a detected object is displayed with an incorrect name, verify the class order against the model's original label list before changing the detection logic.

⚙️ Configuration
Setting	Where to change it	Current behavior
Confidence threshold	real_time_object_detection.py	Defaults to 0.2; configurable with --confidence
Camera device	app.py and real_time_object_detection.py	Uses camera index 0
Input size	Detection scripts	Resizes model input to 300 × 300
Frame width	Detection scripts	Resizes output frames to a maximum width of approximately 800 pixels
For better results, use a well-lit environment, keep the camera steady, and ensure objects are visible in the frame.

🛠️ Troubleshooting
Webcam is not opening

Check that your camera is connected and available.
Close other applications that may be using the camera.
Check the camera index. The current implementation uses 0; another device may require a different index.
Model files are missing

Make sure both files are present in the project root:

MobileNetSSD_deploy.prototxt.txt
MobileNetSSD_deploy.caffemodel
ModuleNotFoundError

Activate your virtual environment and install the required dependencies:

pip install opencv-python numpy Flask imutils
The web page opens but no video appears

Select Start in the browser interface.
Check the Flask terminal for camera or model-loading errors.
Verify that the browser is accessing the same machine on which Flask is running.
Detection is slow or inconsistent

Performance depends on your CPU, camera resolution, lighting, and scene complexity. Close other CPU-intensive applications and try a lower camera resolution.

🔬 Related Work: Tiny Object Detection for Microscopy
This project focuses on detecting everyday objects from a live webcam using MobileNet SSD. A related computer-vision project explores a different use case: detecting tiny biological structures in microscopic images.

For reference, see Tiny Object Detector – Microscopic by PRATHAM-p25. That repository describes a Streamlit-based microscopy application using a trained YOLO model exported to ONNX, with image annotation and object counting. It also documents optional result persistence through MongoDB Atlas and GridFS.

How the projects relate
Area	This repository	Microscopy reference
Main use case	General object detection from a webcam	Tiny biological object detection in microscope images
Detection model	MobileNet SSD (Caffe)	YOLO exported to ONNX
Application interface	Flask with HTML, CSS, and JavaScript	Streamlit
Input	Live webcam frames	Uploaded or captured microscope images
Result handling	Live annotated video stream	Annotated images, object counts, and documented cloud storage
The microscopy repository is included as a related project and reference, not as a dependency or component of this codebase. The two projects use different models and application workflows, but both demonstrate how computer vision can automate object detection and visual analysis in different domains.

🔭 Future improvements
Potential directions for extending this project include:

Add image and prerecorded-video upload for offline detection.
Display detection counts and a live object summary.
Add adjustable confidence controls to the web interface.
Improve camera selection and error handling.
Add detection history and exportable reports.
Explore optimized inference using ONNX or TensorFlow Lite.
Improve the frontend with a responsive dashboard.
🤝 Contributing
Contributions, suggestions, and bug reports are welcome.

Fork the repository.
Create a feature branch:
git checkout -b feature/your-feature
Make your changes and test them locally.
Commit your changes:
git commit -m "Add: your feature"
Push your branch and open a Pull Request.
📄 License
This project includes a LICENSE file. Please review it for the applicable terms before using, modifying, or distributing the project.

👨‍💻 Author
Ravi Kumar N Rao

GitHub: @ravirao10
LinkedIn: Ravi Kumar N Rao
HackerRank: ravikumarrao2004
<p align="center"> <strong>Built to explore computer vision, real-time inference, and web-based AI applications.</strong> <br> If you find this project useful, consider giving the repository a ⭐ </p>
