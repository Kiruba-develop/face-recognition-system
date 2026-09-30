import os
import pickle

import cv2
import face_recognition
import numpy as np


class FaceRecognizer:
    """Face recognition using the face_recognition library and OpenCV."""

    def __init__(self, encodings_path='data/known_faces/encodings.pkl'):
        self.encodings_path = encodings_path
        self.known_face_names = []
        self.known_face_encodings = []
        self.load_known_faces()

    def _ensure_dataset_exists(self):
        os.makedirs(os.path.dirname(self.encodings_path), exist_ok=True)

    def load_known_faces(self):
        self._ensure_dataset_exists()
        if os.path.exists(self.encodings_path):
            with open(self.encodings_path, 'rb') as f:
                data = pickle.load(f)
                self.known_face_names = data.get('names', [])
                self.known_face_encodings = data.get('encodings', [])

    def save_known_faces(self):
        self._ensure_dataset_exists()
        with open(self.encodings_path, 'wb') as f:
            pickle.dump({
                'names': self.known_face_names,
                'encodings': self.known_face_encodings,
            }, f)

    def add_face(self, image_path, name):
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if not encodings:
            raise ValueError(f'No face detected in {image_path}')
        self.known_face_names.append(name)
        self.known_face_encodings.append(encodings[0])
        self.save_known_faces()

    def add_faces_from_directory(self, directory):
        for person_name in os.listdir(directory):
            person_dir = os.path.join(directory, person_name)
            if not os.path.isdir(person_dir):
                continue
            for file_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, file_name)
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    try:
                        self.add_face(image_path, person_name)
                    except ValueError:
                        continue

    def recognize_image(self, image_path, tolerance=0.45):
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        results = []
        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(
                self.known_face_encodings,
                face_encoding,
                tolerance=tolerance,
            )
            if True in matches:
                match_index = matches.index(True)
                label = self.known_face_names[match_index]
                distance = face_recognition.face_distance(
                    [self.known_face_encodings[match_index]],
                    face_encoding,
                )[0]
                confidence = max(0.0, 1.0 - distance)
            else:
                label = 'Unknown'
                confidence = 0.0
            results.append((label, confidence, face_location))
        return results

    def recognize_frame(self, frame, tolerance=0.45):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        results = []
        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(
                self.known_face_encodings,
                face_encoding,
                tolerance=tolerance,
            )
            if True in matches:
                match_index = matches.index(True)
                label = self.known_face_names[match_index]
                distance = face_recognition.face_distance(
                    [self.known_face_encodings[match_index]],
                    face_encoding,
                )[0]
                confidence = max(0.0, 1.0 - distance)
            else:
                label = 'Unknown'
                confidence = 0.0
            results.append({
                'name': label,
                'confidence': confidence,
                'location': face_location,
            })
        return results

    def verify_face(self, image_path_1, image_path_2, tolerance=0.45):
        image1 = face_recognition.load_image_file(image_path_1)
        image2 = face_recognition.load_image_file(image_path_2)

        enc1 = face_recognition.face_encodings(image1)
        enc2 = face_recognition.face_encodings(image2)

        if not enc1 or not enc2:
            return False, 0.0

        distance = face_recognition.face_distance([enc1[0]], enc2[0])[0]
        same = distance <= tolerance
        return same, 1.0 - distance
