# Face Recognition System

A real-time face recognition project built with Python and OpenCV. The system detects faces through a webcam and compares them with registered face images to identify known individuals.

## Features

- Real-time face detection using OpenCV YuNet
- Face feature extraction and matching using OpenCV SFace
- Webcam-based face registration
- Recognition of registered faces
- Unknown-face handling
- Modular Python project structure

## Tech Stack

- Python 3.12
- OpenCV
- NumPy
- YuNet face detector
- SFace face recognizer

## Project Structure

```text
face-recognition-system/
├── models/
│   ├── face_detection_yunet_2023mar.onnx
│   └── face_recognition_sface_2021dec.onnx
├── data/
│   └── registered_faces/
├── src/
│   ├── detection.py
│   ├── main.py
│   ├── recognition.py
│   └── register_face.py
├── tests/
├── evaluation/
├── docs/
├── requirements.txt
├── .gitignore
└── README.md
```

## Requirements

- Python 3.12
- A webcam
- Git
- Internet access to download the pretrained models

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/om-srivastav/face-recognition-system.git
cd face-recognition-system
```

### 2. Create and activate a virtual environment

Windows CMD:

```cmd
py -3.12 -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```cmd
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Download Pretrained Models

Download the following models and place them inside the `models/` directory.

- **YuNet:** [OpenCV Zoo – Face Detection YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
- **SFace:** [OpenCV Zoo – Face Recognition SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface)

Use these filenames:

```text
models/face_detection_yunet_2023mar.onnx
models/face_recognition_sface_2021dec.onnx
```

Ensure the downloaded files are complete and match the filenames expected by the Python scripts.

## Usage

### Run Face Detection

```cmd
python src\main.py
```

### Register a Face

```cmd
python src\register_face.py
```

Enter a person's name when prompted. Keep one face clearly visible in the webcam and press **S** to save the registration photo. Press **Q** to cancel.

### Run Face Recognition

```cmd
python src\recognition.py
```

The webcam will open and compare detected faces against the registered face images. Press **Q** to exit.

## Registering Faces

Registration photos are stored locally in:

```text
data/registered_faces/
```

Keep this directory private. Do not commit personal face images or other biometric data to a public repository.

## Limitations

- Recognition accuracy depends on lighting, camera angle, image quality, and registration photos.
- The similarity threshold is a configurable starting value, not a confidence percentage.
- Unknown-face detection and matching thresholds need testing and evaluation.
- This project is a prototype and should not be used as the sole basis for security-sensitive decisions.

## Future Improvements

- Multiple registration images per person
- Better registration validation
- Automated accuracy evaluation
- Unit and integration tests
- Improved user interface
- Performance and threshold tuning

## Privacy

Obtain permission before collecting or processing anyone's face images. Store biometric data securely and delete it when no longer required.

## Author

**Om Srivastav**

GitHub: [@om-srivastav](https://github.com/om-srivastav)

## License

Add a suitable open-source license before redistributing this project.
