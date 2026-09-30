"""OpenCV-only Face Recognition Implementation"""

import os
import cv2
import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Optional


class OpenCVFaceRecognizer:
    """
    Pure OpenCV-based face recognition using:
    - Haar Cascade Classifiers for face detection
    - LBPH (Local Binary Patterns Histograms) for face recognition
    - No external face_recognition dependency
    """

    def __init__(self, encodings_path="data/known_faces/lbph_model.yml"):
        self.encodings_path = encodings_path
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.known_faces = {}
        self.label_to_name = {}
        self.next_label = 0
        self.load_model()

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using Haar Cascade."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return faces

    def preprocess_face(self, face: np.ndarray, size=(200, 200)) -> np.ndarray:
        """Preprocess face for LBPH recognition."""
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, size)
        return face

    def train_from_directory(self, dataset_path: str):
        """Train LBPH recognizer from directory structure.
        
        Expected structure:
            dataset/
            ├── person1/
            │   ├── img1.jpg
            │   └── img2.jpg
            └── person2/
                └── img1.jpg
        """
        faces = []
        labels = []

        for person_name in os.listdir(dataset_path):
            person_dir = os.path.join(dataset_path, person_name)
            if not os.path.isdir(person_dir):
                continue

            label = self.next_label
            self.label_to_name[label] = person_name
            self.next_label += 1

            for filename in os.listdir(person_dir):
                if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    image_path = os.path.join(person_dir, filename)
                    image = cv2.imread(image_path)
                    if image is None:
                        continue

                    face = self.detect_faces(image)
                    if len(face) > 0:
                        x, y, w, h = face[0]
                        face_roi = image[y : y + h, x : x + w]
                        preprocessed = self.preprocess_face(face_roi)
                        faces.append(preprocessed)
                        labels.append(label)

        if faces:
            self.face_recognizer.train(faces, np.array(labels))
            self.save_model()
            print(f"Trained LBPH recognizer with {len(faces)} images")
        else:
            print("No valid faces found for training")

    def recognize_face(self, image: np.ndarray, confidence_threshold=50):
        """Recognize faces in an image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detect_faces(image)
        results = []

        for x, y, w, h in faces:
            face_roi = gray[y : y + h, x : x + w]
            face_roi = cv2.resize(face_roi, (200, 200))

            label, confidence = self.face_recognizer.predict(face_roi)
            name = self.label_to_name.get(label, "Unknown")
            confidence_score = 100 - confidence

            results.append({
                "name": name,
                "confidence": confidence_score,
                "location": (x, y, w, h),
                "label": label,
            })

        return results

    def save_model(self):
        """Save trained model."""
        os.makedirs(os.path.dirname(self.encodings_path), exist_ok=True)
        self.face_recognizer.save(self.encodings_path)
        metadata = {
            "label_to_name": self.label_to_name,
            "next_label": self.next_label,
        }
        with open(self.encodings_path.replace(".yml", "_meta.pkl"), "wb") as f:
            pickle.dump(metadata, f)

    def load_model(self):
        """Load trained model."""
        if os.path.exists(self.encodings_path):
            self.face_recognizer.read(self.encodings_path)
            meta_path = self.encodings_path.replace(".yml", "_meta.pkl")
            if os.path.exists(meta_path):
                with open(meta_path, "rb") as f:
                    metadata = pickle.load(f)
                    self.label_to_name = metadata.get("label_to_name", {})
                    self.next_label = metadata.get("next_label", 0)


if __name__ == "__main__":
    recognizer = OpenCVFaceRecognizer()
    print("OpenCV Face Recognizer initialized")
