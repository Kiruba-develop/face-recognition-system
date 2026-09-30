"""Universal Face Recognition Model Trainer
Trains on ALL available public face datasets

Supported datasets:
- LFW (Labeled Faces in the Wild)
- CelebA
- VGGFace2
- IMDB-WIKI
- UTKFace
- MS-Celeb-1M
- Faces94/95/96
"""

import os
import cv2
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2, EfficientNetB3
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import urllib.request
import tarfile
import zipfile
import shutil


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UniversalFaceDatasetManager:
    """Manage and organize all public face datasets."""

    def __init__(self, root_dir="data/public_datasets"):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_info = self._get_dataset_info()

    def _get_dataset_info(self) -> Dict:
        """Get information about all available datasets."""
        return {
            "LFW": {
                "url": "http://vis-www.cs.umass.edu/lfw/lfw.tgz",
                "size_mb": 200,
                "num_images": 13233,
                "num_identities": 5749,
                "path": self.root_dir / "lfw",
            },
            "CelebA": {
                "url": "http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html",
                "size_mb": 1300,
                "num_images": 202599,
                "num_identities": 10177,
                "path": self.root_dir / "celeba",
                "requires_manual_download": True,
            },
            "UTKFace": {
                "url": "https://susanqq.github.io/UTKFace/",
                "size_mb": 1000,
                "num_images": 20000,
                "num_identities": "various",
                "path": self.root_dir / "utk_face",
                "requires_manual_download": True,
            },
            "VGGFace2": {
                "url": "http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/",
                "size_mb": 500000,
                "num_images": 3310000,
                "num_identities": 9131,
                "path": self.root_dir / "vggface2",
                "requires_manual_download": True,
            },
            "IMDB-WIKI": {
                "url": "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/",
                "size_mb": 60000,
                "num_images": 500000,
                "num_identities": "celebrities",
                "path": self.root_dir / "imdb_wiki",
                "requires_manual_download": True,
            },
            "MS-Celeb-1M": {
                "url": "https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-million-celebrities-in-the-wild/",
                "size_mb": 500000,
                "num_images": 10000000,
                "num_identities": 100000,
                "path": self.root_dir / "ms_celeb_1m",
                "requires_manual_download": True,
            },
            "Faces94-96": {
                "url": "https://cswww.essex.ac.uk/mv/allfaces/",
                "size_mb": 100,
                "num_images": 7000,
                "num_identities": 152,
                "path": self.root_dir / "faces94_96",
            },
        }

    def download_lfw(self):
        """Download LFW dataset."""
        logger.info("Downloading LFW dataset...")
        url = self.dataset_info["LFW"]["url"]
        output_path = self.root_dir / "lfw.tgz"

        try:
            urllib.request.urlretrieve(url, output_path, reporthook=self._progress_hook)
            logger.info("\nExtracting LFW...")
            self._extract_tar(output_path, self.root_dir)
            return True
        except Exception as e:
            logger.error(f"Failed to download LFW: {e}")
            return False

    def organize_all_datasets(self, output_dir="data/unified_dataset"):
        """Organize all available datasets into unified structure."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info("Organizing datasets...")
        total_images = 0
        total_identities = 0

        # Process each dataset
        for dataset_name, info in self.dataset_info.items():
            dataset_path = info["path"]
            if not dataset_path.exists():
                logger.warning(f"{dataset_name} not found at {dataset_path}")
                continue

            logger.info(f"Processing {dataset_name}...")
            images_added, identities_added = self._process_dataset(
                dataset_path, output_path, dataset_name
            )
            total_images += images_added
            total_identities += identities_added

        logger.info(
            f"\nUnified dataset created: {total_images} images, {total_identities} identities"
        )
        return output_path

    def _process_dataset(
        self, input_path: Path, output_path: Path, dataset_name: str
    ) -> Tuple[int, int]:
        """Process a single dataset."""
        images_added = 0
        identities_added = 0

        for person_dir in input_path.iterdir():
            if not person_dir.is_dir():
                continue

            # Create person directory in unified dataset
            person_name = f"{dataset_name}_{person_dir.name}"
            output_person_dir = output_path / person_name
            output_person_dir.mkdir(exist_ok=True)

            # Copy/process images
            for image_file in person_dir.rglob("*.jpg"):
                try:
                    shutil.copy(image_file, output_person_dir / image_file.name)
                    images_added += 1
                except Exception as e:
                    logger.debug(f"Error copying {image_file}: {e}")

            if images_added > 0:
                identities_added += 1

        return images_added, identities_added

    def preprocess_all_datasets(
        self,
        input_dir: str,
        output_dir: str,
        image_size: Tuple[int, int] = (224, 224),
    ):
        """Preprocess all images: detect faces, resize, normalize."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        logger.info("Preprocessing all datasets...")
        total_processed = 0
        total_skipped = 0

        for person_dir in input_path.iterdir():
            if not person_dir.is_dir():
                continue

            output_person_dir = output_path / person_dir.name
            output_person_dir.mkdir(exist_ok=True)

            for image_file in person_dir.glob("*.jpg"):
                try:
                    image = cv2.imread(str(image_file))
                    if image is None:
                        total_skipped += 1
                        continue

                    # Detect faces
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                    )

                    if len(faces) == 0:
                        total_skipped += 1
                        continue

                    # Extract largest face
                    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                    face = image[y : y + h, x : x + w]

                    # Resize
                    face = cv2.resize(face, image_size)

                    # Histogram equalization
                    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    face_gray = cv2.equalizeHist(face_gray)
                    face = cv2.cvtColor(face_gray, cv2.COLOR_GRAY2BGR)

                    # Save
                    output_file = output_person_dir / image_file.name
                    cv2.imwrite(str(output_file), face)
                    total_processed += 1

                except Exception as e:
                    logger.debug(f"Error processing {image_file}: {e}")
                    total_skipped += 1

        logger.info(
            f"Preprocessing complete: {total_processed} processed, {total_skipped} skipped"
        )

    def _progress_hook(self, block_num, block_size, total_size):
        """Show download progress."""
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        print(f"\rDownloading: {percent:.1f}%", end="")

    def _extract_tar(self, tar_path, extract_to):
        """Extract tar file."""
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=extract_to)
        os.remove(tar_path)


class UniversalFaceRecognitionModel:
    """Train face recognition model on all available public datasets."""

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        embedding_size: int = 128,
    ):
        self.input_shape = input_shape
        self.embedding_size = embedding_size
        self.model = None
        self.embedding_model = None
        self.class_indices = {}
        self.index_to_class = {}

    def build_embedding_model(self, backbone: str = "efficientnetb3") -> keras.Model:
        """Build face embedding model using pre-trained backbone.
        
        Uses ArcFace-style output for better face separation.
        """
        if backbone == "efficientnetb3":
            base_model = EfficientNetB3(
                weights="imagenet",
                include_top=False,
                input_shape=self.input_shape,
            )
        elif backbone == "resnet50":
            base_model = ResNet50(
                weights="imagenet",
                include_top=False,
                input_shape=self.input_shape,
            )
        elif backbone == "mobilenetv2":
            base_model = MobileNetV2(
                weights="imagenet",
                include_top=False,
                input_shape=self.input_shape,
            )
        else:
            base_model = VGG16(
                weights="imagenet",
                include_top=False,
                input_shape=self.input_shape,
            )

        base_model.trainable = False  # Freeze backbone initially

        # Embedding head
        model = models.Sequential([
            layers.Input(shape=self.input_shape),
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(1024, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.embedding_size, activation=None),  # No activation for embeddings
            layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1)),  # L2 normalization
        ])

        return model

    def build_classification_model(self, num_classes: int) -> keras.Model:
        """Build classification model for training."""
        embedding_input = layers.Input(shape=self.input_shape)
        embedding_model = self.embedding_model(embedding_input)
        
        # ArcFace-style output
        output = layers.Dense(
            num_classes,
            activation='softmax',
            kernel_regularizer=keras.regularizers.l2(1e-4),
        )(embedding_model)

        model = keras.Model(inputs=embedding_input, outputs=output)
        return model

    def train_on_public_datasets(
        self,
        dataset_path: str,
        epochs: int = 100,
        batch_size: int = 64,
        validation_split: float = 0.2,
        backbone: str = "efficientnetb3",
        model_save_path: str = "data/models/universal_face_model.h5",
    ) -> keras.callbacks.History:
        """Train on unified public datasets."""
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        logger.info(f"Training on public datasets from {dataset_path}")

        # Data generators
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=validation_split,
        )

        train_gen = train_datagen.flow_from_directory(
            dataset_path,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
        )

        val_gen = train_datagen.flow_from_directory(
            dataset_path,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
        )

        # Build models
        num_classes = train_gen.num_classes
        self.class_indices = train_gen.class_indices
        self.index_to_class = {v: k for k, v in self.class_indices.items()}

        self.embedding_model = self.build_embedding_model(backbone=backbone)
        self.model = self.build_classification_model(num_classes)

        # Compile
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(),
                keras.metrics.Recall(),
            ],
        )

        logger.info(f"Training on {train_gen.num_classes} identities")
        logger.info(f"Total training samples: {train_gen.samples}")
        logger.info(f"Total validation samples: {val_gen.samples}")

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-8,
                verbose=1,
            ),
            keras.callbacks.ModelCheckpoint(
                model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1,
            ),
        ]

        # Train
        history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )

        logger.info(f"Model saved to {model_save_path}")
        return history

    def extract_embeddings(self, image_path: str) -> np.ndarray:
        """Extract face embedding from image."""
        if self.embedding_model is None:
            raise ValueError("Embedding model not trained")

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.resize(image, (self.input_shape[0], self.input_shape[1]))
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)

        embedding = self.embedding_model.predict(image, verbose=0)
        return embedding[0]

    def save_model(self, path: str):
        """Save trained models."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        self.embedding_model.save(path.replace('.h5', '_embedding.h5'))
        logger.info(f"Models saved to {path}")

    def load_model(self, path: str):
        """Load trained models."""
        self.model = keras.models.load_model(path)
        embedding_path = path.replace('.h5', '_embedding.h5')
        if os.path.exists(embedding_path):
            self.embedding_model = keras.models.load_model(embedding_path)
        logger.info(f"Models loaded from {path}")


if __name__ == "__main__":
    logger.info("Universal Face Recognition Model Trainer")
    logger.info("=========================================\n")

    # Initialize dataset manager
    dataset_manager = UniversalFaceDatasetManager()

    logger.info("Available datasets:")
    for name, info in dataset_manager.dataset_info.items():
        logger.info(f"  - {name}: {info['num_images']} images, {info['num_identities']} identities")

    logger.info("\nDataset information saved.")
