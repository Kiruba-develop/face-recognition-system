import cv2
import numpy as np


class FaceDetector:
    """Wrapper around OpenCV Haar cascade face detection."""

    def __init__(self, cascade_path=None, scale_factor=1.1, min_neighbors=5):
        if cascade_path is None:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

    def detect_faces(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(30, 30),
        )
        return faces

    def draw_boxes(self, image, faces):
        output = image.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return output

    def crop_faces(self, image, faces):
        crops = []
        for (x, y, w, h) in faces:
            crop = image[y:y + h, x:x + w]
            crops.append(crop)
        return crops
