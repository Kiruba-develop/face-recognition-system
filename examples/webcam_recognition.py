"""Real-time webcam face recognition

Runs recognition on live webcam feed.

Features:
- Real-time face detection and recognition
- Displays identity and confidence score
- Shows FPS counter
- Press 'q' to quit
- Press 's' to save detected faces

Usage:
    python examples/webcam_recognition.py --model data/models/custom_face_model.h5 \\
                                          --database data/models/custom_identity_db.pkl

Keybindings:
    q - Quit
    s - Save detected face
    space - Pause/Resume
    t - Toggle detection on/off
"""

import cv2
import numpy as np
import logging
import argparse
from pathlib import Path
from time import time
from src.face_embedding_model import FaceEmbeddingModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebcamFaceRecognizer:
    """Real-time face recognition on webcam."""
    
    def __init__(
        self,
        model_path: str,
        database_path: str,
        confidence_threshold: float = 0.6,
        display_fps: bool = True,
    ):
        logger.info("Loading models...")
        
        self.model = FaceEmbeddingModel()
        self.model.load_models(model_path)
        self.model.load_identity_database(database_path)
        
        self.confidence_threshold = confidence_threshold
        self.display_fps = display_fps
        self.detected_faces = []
        self.frame_count = 0
        self.fps = 0
        self.last_time = time()
        self.paused = False
        self.detection_enabled = True
        
        # Face cascade for detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        logger.info(f"✅ Models loaded")
        logger.info(f"   Identity database: {len(self.model.identity_database)} identities")
        logger.info(f"   Confidence threshold: {confidence_threshold}")
    
    def detect_faces(self, frame: np.ndarray):
        """Detect faces in frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50),
        )
        return faces
    
    def recognize_face(self, face_region: np.ndarray) -> tuple:
        """Recognize face in region."""
        try:
            # Save temporarily
            temp_path = "/tmp/face_temp.jpg"
            cv2.imwrite(temp_path, face_region)
            
            # Recognize
            results = self.model.recognize_face(
                temp_path,
                threshold=self.confidence_threshold,
                top_k=1,
            )
            
            if results and results[0][0] != "Unknown":
                return results[0]
            else:
                return ("Unknown", 0.0)
        except Exception as e:
            logger.debug(f"Recognition error: {e}")
            return ("Unknown", 0.0)
    
    def draw_results(
        self,
        frame: np.ndarray,
        faces: np.ndarray,
        identities: list,
    ) -> np.ndarray:
        """Draw detection boxes and labels."""
        for (x, y, w, h), (name, conf) in zip(faces, identities):
            # Draw box
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"{name} ({conf:.2f})" if conf > 0 else name
            cv2.putText(
                frame,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
            
            # Draw confidence bar
            bar_width = w
            bar_height = 5
            bar_y = y + h + 5
            cv2.rectangle(frame, (x, bar_y), (x + int(bar_width * conf), bar_y + bar_height), color, -1)
            cv2.rectangle(frame, (x, bar_y), (x + bar_width, bar_y + bar_height), color, 2)
        
        return frame
    
    def draw_ui(self, frame: np.ndarray) -> np.ndarray:
        """Draw UI elements."""
        height, width = frame.shape[:2]
        
        # FPS
        if self.display_fps:
            fps_text = f"FPS: {self.fps:.1f}"
            cv2.putText(
                frame,
                fps_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
        
        # Status
        status_text = "PAUSED" if self.paused else ("DETECTION ON" if self.detection_enabled else "DETECTION OFF")
        status_color = (0, 0, 255) if self.paused else ((0, 255, 0) if self.detection_enabled else (0, 165, 255))
        cv2.putText(
            frame,
            status_text,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2,
        )
        
        # Instructions
        instructions = [
            "Press q: Quit",
            "Press s: Save face",
            "Press space: Pause",
            "Press t: Toggle detection",
        ]
        y_offset = height - 100
        for i, instruction in enumerate(instructions):
            cv2.putText(
                frame,
                instruction,
                (10, y_offset + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )
        
        return frame
    
    def run(self, camera_id: int = 0):
        """Run webcam recognition."""
        logger.info("\n" + "="*80)
        logger.info("WEBCAM FACE RECOGNITION")
        logger.info("="*80)
        logger.info("\nStarting webcam...")
        logger.info("Press 'q' to quit")
        logger.info("Press 's' to save detected face")
        logger.info("Press space to pause/resume")
        logger.info("Press 't' to toggle detection")
        
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            logger.error("Could not open webcam")
            return
        
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Update FPS
                current_time = time()
                self.fps = 1.0 / (current_time - self.last_time + 1e-6)
                self.last_time = current_time
                
                # Detect and recognize
                identities = []
                faces = np.array([])
                
                if not self.paused and self.detection_enabled:
                    faces = self.detect_faces(frame)
                    
                    for (x, y, w, h) in faces:
                        face_region = frame[y:y+h, x:x+w]
                        name, conf = self.recognize_face(face_region)
                        identities.append((name, conf))
                        self.detected_faces.append({
                            'name': name,
                            'confidence': conf,
                            'frame': face_region.copy(),
                            'timestamp': current_time,
                        })
                
                # Draw
                if len(identities) > 0:
                    frame = self.draw_results(frame, faces, identities)
                
                frame = self.draw_ui(frame)
                
                # Display
                cv2.imshow("Face Recognition", frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    logger.info("\nQuitting...")
                    break
                elif key == ord('s'):
                    if self.detected_faces:
                        face_data = self.detected_faces[-1]
                        filename = f"detected_face_{face_data['name']}_{int(face_data['timestamp'])}.jpg"
                        cv2.imwrite(filename, face_data['frame'])
                        logger.info(f"Saved: {filename}")
                elif key == ord(' '):
                    self.paused = not self.paused
                    status = "PAUSED" if self.paused else "RESUMED"
                    logger.info(f"{status}")
                elif key == ord('t'):
                    self.detection_enabled = not self.detection_enabled
                    status = "ON" if self.detection_enabled else "OFF"
                    logger.info(f"Detection: {status}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("\nWebcam closed")
            logger.info(f"Total faces detected: {len(self.detected_faces)}")


def main():
    parser = argparse.ArgumentParser(description="Real-time face recognition on webcam")
    parser.add_argument(
        "--model",
        type=str,
        default="data/models/custom_face_model.h5",
        help="Path to embedding model",
    )
    parser.add_argument(
        "--database",
        type=str,
        default="data/models/custom_identity_db.pkl",
        help="Path to identity database",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Confidence threshold (0-1)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera ID (default: 0)",
    )
    parser.add_argument(
        "--no-fps",
        action="store_true",
        help="Don't display FPS",
    )
    
    args = parser.parse_args()
    
    # Check files exist
    if not Path(args.model).exists():
        logger.error(f"Model not found: {args.model}")
        logger.error(f"\nTrain a model first:")
        logger.error(f"  python examples/quick_train_custom_only.py")
        return
    
    if not Path(args.database).exists():
        logger.error(f"Database not found: {args.database}")
        return
    
    # Run
    recognizer = WebcamFaceRecognizer(
        model_path=args.model,
        database_path=args.database,
        confidence_threshold=args.threshold,
        display_fps=not args.no_fps,
    )
    
    recognizer.run(camera_id=args.camera)


if __name__ == "__main__":
    main()
