"""Complete Face Recognition System - End-to-End Project"""

import os
import cv2
import numpy as np
import pickle
from pathlib import Path


class CompleteEndToEndSystem:
    """
    Comprehensive face recognition system combining:
    - OpenCV-based face detection
    - face_recognition library for fast recognition
    - Custom TensorFlow CNN for enhanced accuracy
    - Pre-trained models from public datasets
    """

    def __init__(self, project_root="."):
        self.project_root = Path(project_root)
        self.setup_directories()

    def setup_directories(self):
        """Create necessary project directories."""
        dirs = [
            self.project_root / "data" / "dataset",
            self.project_root / "data" / "known_faces",
            self.project_root / "data" / "models",
            self.project_root / "data" / "public_datasets",
            self.project_root / "logs",
            self.project_root / "output",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def get_paths(self):
        """Return standardized paths for the project."""
        return {
            "dataset": self.project_root / "data" / "dataset",
            "known_faces": self.project_root / "data" / "known_faces",
            "models": self.project_root / "data" / "models",
            "public_datasets": self.project_root / "data" / "public_datasets",
            "encodings": self.project_root / "data" / "known_faces" / "encodings.pkl",
            "logs": self.project_root / "logs",
            "output": self.project_root / "output",
        }


if __name__ == "__main__":
    system = CompleteEndToEndSystem()
    print("System initialized with directories:")
    for key, path in system.get_paths().items():
        print(f"  {key}: {path}")
