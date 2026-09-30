import os
import pickle
from typing import Dict, List


class FaceDatabase:
    """Simple database wrapper for storing face names and encodings."""

    def __init__(self, path='data/known_faces/encodings.pkl'):
        self.path = path
        self.data = {'names': [], 'encodings': []}
        self.load()

    def load(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if os.path.exists(self.path):
            with open(self.path, 'rb') as f:
                loaded = pickle.load(f)
                self.data = loaded if isinstance(loaded, dict) else {'names': [], 'encodings': []}

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, 'wb') as f:
            pickle.dump(self.data, f)

    def add_face(self, name: str, encoding):
        self.data['names'].append(name)
        self.data['encodings'].append(encoding)
        self.save()

    def add_faces_from_directory(self, directory: str):
        for person in os.listdir(directory):
            person_dir = os.path.join(directory, person)
            if not os.path.isdir(person_dir):
                continue
            for image_file in os.listdir(person_dir):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    image_path = os.path.join(person_dir, image_file)
                    # This method assumes the calling code will encode the face before saving.
                    # It is left intentionally simple for extension.
                    print(f'Prepared face entry for {person}: {image_path}')

    def export_encodings(self, export_path: str):
        with open(export_path, 'wb') as f:
            pickle.dump(self.data, f)

    def get_names(self):
        return self.data.get('names', [])

    def get_encodings(self):
        return self.data.get('encodings', [])

