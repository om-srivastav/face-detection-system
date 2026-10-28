import cv2
from detection import detect_faces



def main():
    print("Starting face detection...")

    camera = cv2.VideoCapture(0,cv2.CAP_DSHOW)

    if not camera.isOpened():
        print("Error: Could not access the webcam. ")
        return
    
    print("Webcam opened! Press Q to quit.")


    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Error: Could not read camera frame.")
                break

            output,count = detect_faces(frame)

            cv2.putText(
                output,
                f"Faces detected : {count}",
                (20,35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0,255,0),
                2,
            )

            cv2.imshow("Face Detection - Prototype",output)


            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break


    except Exception as error:
        print("ERROR",repr(error))

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Camera released.")



if __name__ == "__main__":
    main()
