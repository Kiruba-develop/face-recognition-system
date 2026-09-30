from src.face_recognizer import FaceRecognizer


def main():
    recognizer = FaceRecognizer()
    results = recognizer.recognize_image('data/test_images/sample.jpg')
    for name, confidence, _ in results:
        print(f'{name}: {confidence:.2f}')


if __name__ == '__main__':
    main()

