
import cv2
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
FACES_DIR = os.path.join(BASE_DIR, "data", "registered_faces")

YUNET = os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx")
SFACE = os.path.join(MODEL_DIR, "face_recognition_sface_2021dec.onnx")

os.makedirs(FACES_DIR, exist_ok=True)

detector = cv2.FaceDetectorYN.create(
    YUNET, "", (320, 320), 0.9, 0.3, 5000
)
recognizer = cv2.FaceRecognizerSF.create(SFACE, "")

registered_faces = []

# Load registered face images
for filename in os.listdir(FACES_DIR):
    path = os.path.join(FACES_DIR, filename)

    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image = cv2.imread(path)
    if image is None:
        continue

    h, w = image.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(image)

    if faces is None or len(faces) == 0:
        print(f"No face found in: {filename}")
        continue

    face = max(faces, key=lambda f: f[2] * f[3])
    aligned = recognizer.alignCrop(image, face)
    feature = recognizer.feature(aligned)

    name = os.path.splitext(filename)[0]
    registered_faces.append((name, feature))
    print(f"Registered: {name}")

if not registered_faces:
    print("No registered faces found!")
    print("Add a clear face photo to data/registered_faces first.")

# Start webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Cannot open webcam.")
    raise SystemExit

print("Face recognition started. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)

    if faces is not None:
        for face in faces:
            aligned = recognizer.alignCrop(frame, face)
            feature = recognizer.feature(aligned)

            best_name = "Unknown"
            best_score = -1

            for name, registered_feature in registered_faces:
                score = recognizer.match(
                    feature, registered_feature,
                    cv2.FaceRecognizerSF_FR_COSINE
                )

                if score > best_score:
                    best_score = score
                    best_name = name

            # Conservative starting threshold; tune using test photos
            if best_score < 0.45:
                best_name = "Unknown"

            x, y, fw, fh = map(int, face[:4])
            color = (0, 200, 0) if best_name != "Unknown" else (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 2)
            label = f"{best_name} ({best_score:.2f})"
            cv2.putText(
                frame, label, (x, max(25, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )

    cv2.imshow("Face Recognition System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
