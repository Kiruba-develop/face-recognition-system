import cv2

from src.face_detector import FaceDetector
from src.face_recognizer import FaceRecognizer


def main():
    detector = FaceDetector()
    recognizer = FaceRecognizer()
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        faces = detector.detect_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        results = recognizer.recognize_frame(frame)
        for result in results:
            name = result['name']
            confidence = result['confidence']
            (top, right, bottom, left) = result['location']
            cv2.putText(
                frame,
                f'{name} {confidence:.2f}',
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

