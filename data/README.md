import os
import unittest

from src.face_recognizer import FaceRecognizer


class TestFaceRecognition(unittest.TestCase):
    def test_known_faces_file_can_load(self):
        recognizer = FaceRecognizer()
        self.assertIsInstance(recognizer.known_face_names, list)
        self.assertIsInstance(recognizer.known_face_encodings, list)


if __name__ == '__main__':
    unittest.main()

