
import cv2
import os
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACES_DIR = os.path.join(BASE_DIR, "data", "registered_faces")
MODEL_PATH = os.path.join(
    BASE_DIR, "models", "face_detection_yunet_2023mar.onnx"
)

os.makedirs(FACES_DIR, exist_ok=True)

name = input("Enter person's name: ").strip()

if not name or not re.fullmatch(r"[\w -]+", name):
    print("Invalid name.")
    raise SystemExit

filename = re.sub(r"\s+", "_", name) + ".jpg"
save_path = os.path.join(FACES_DIR, filename)

if os.path.exists(save_path):
    print("This person is already registered. Use another name.")
    raise SystemExit

detector = cv2.FaceDetectorYN.create(
    MODEL_PATH, "", (320, 320), 0.9, 0.3, 5000
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Cannot open webcam. Close other apps using the camera.")
    raise SystemExit

print("Keep exactly one face visible.")
print("Press S to save, Q to cancel.")

saved = False

while True:
    ret, frame = cap.read()

    if not ret:
        print("Cannot read webcam frame.")
        break

    height, width = frame.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(frame)

    display = frame.copy()

    if faces is not None:
        for face in faces:
            x, y, fw, fh = map(int, face[:4])
            cv2.rectangle(
                display, (x, y), (x + fw, y + fh),
                (0, 255, 0), 2
            )

        message = "Face detected - Press S to save"
        color = (0, 255, 0)
    else:
        message = "No face detected"
        color = (0, 0, 255)

    cv2.putText(
        display, message, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2
    )

    cv2.imshow("Register Face", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        if faces is None or len(faces) != 1:
            print("Exactly one face must be visible.")
            continue

        if cv2.imwrite(save_path, frame):
            print("Registration successful:", filename)
            saved = True
        else:
            print("Could not save photo.")
        break

    elif key == ord("q"):
        print("Registration cancelled.")
        break

cap.release()
cv2.destroyAllWindows()

if saved:
    print("Photo saved in:", FACES_DIR)
