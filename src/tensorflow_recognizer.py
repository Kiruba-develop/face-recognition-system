"""TensorFlow/Keras Custom-Trained Face Recognition Model"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from pathlib import Path
from typing import Tuple, Dict, List


class TensorFlowFaceRecognizer:
    """
    Custom TensorFlow/Keras-based face recognition with:
    - Custom CNN architecture option
    - Transfer learning with pre-trained models (VGG16, ResNet50, MobileNetV2)
    - Data augmentation for improved robustness
    - Support for training on public face datasets
    """

    def __init__(self, input_shape=(224, 224, 3), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.class_indices = {}
        self.index_to_class = {}

    def build_custom_cnn(self) -> keras.Model:
        """Build custom CNN architecture for face recognition.
        
        Architecture:
            Input -> Conv2D -> BatchNorm -> ReLU -> MaxPool -> Dropout
                  -> Conv2D -> BatchNorm -> ReLU -> MaxPool -> Dropout
                  -> Conv2D -> BatchNorm -> ReLU -> MaxPool -> Dropout
                  -> Conv2D -> BatchNorm -> ReLU -> MaxPool -> Dropout
                  -> Flatten -> Dense -> Dropout -> Dense -> Softmax
        """
        model = models.Sequential([
            layers.Input(shape=self.input_shape),
            # Block 1
            layers.Conv2D(32, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            # Block 2
            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            # Block 3
            layers.Conv2D(128, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            # Block 4
            layers.Conv2D(256, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            # Dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax'),
        ])
        return model

    def build_transfer_learning_model(
        self, backbone="vgg16", freeze_backbone=True
    ) -> keras.Model:
        """Build transfer learning model using pre-trained backbone.
        
        Args:
            backbone: 'vgg16', 'resnet50', or 'mobilenetv2'
            freeze_backbone: Whether to freeze backbone weights
        """
        if backbone == "vgg16":
            base_model = VGG16(
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
            raise ValueError(f"Unknown backbone: {backbone}")

        if freeze_backbone:
            base_model.trainable = False

        model = models.Sequential([
            layers.Input(shape=self.input_shape),
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax'),
        ])
        return model

    def compile_model(self, learning_rate=0.001):
        """Compile the model."""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()],
        )

    def train(
        self,
        dataset_path: str,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2,
        model_type: str = "custom",
        backbone: str = "mobilenetv2",
    ) -> keras.callbacks.History:
        """Train the model on face dataset.
        
        Args:
            dataset_path: Path to dataset directory with class subdirectories
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            model_type: 'custom' or 'transfer_learning'
            backbone: Backbone for transfer learning ('vgg16', 'resnet50', 'mobilenetv2')
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        # Data generators with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=validation_split,
        )

        train_generator = train_datagen.flow_from_directory(
            dataset_path,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
        )

        validation_generator = train_datagen.flow_from_directory(
            dataset_path,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
        )

        # Store class information
        self.num_classes = train_generator.num_classes
        self.class_indices = train_generator.class_indices
        self.index_to_class = {v: k for k, v in self.class_indices.items()}

        # Build model
        if model_type == "custom":
            self.model = self.build_custom_cnn()
        elif model_type == "transfer_learning":
            self.model = self.build_transfer_learning_model(backbone=backbone)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.compile_model()

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
            ),
        ]

        # Train
        history = self.model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            callbacks=callbacks,
        )

        return history

    def evaluate(self, test_dataset_path: str) -> Dict:
        """Evaluate model on test dataset."""
        if self.model is None:
            raise ValueError("Model not trained yet")

        test_datagen = ImageDataGenerator(rescale=1.0 / 255)
        test_generator = test_datagen.flow_from_directory(
            test_dataset_path,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=32,
            class_mode='categorical',
            shuffle=False,
        )

        predictions = self.model.predict(test_generator)
        predicted_labels = np.argmax(predictions, axis=1)
        true_labels = test_generator.classes

        accuracy = accuracy_score(true_labels, predicted_labels)
        conf_matrix = confusion_matrix(true_labels, predicted_labels)
        class_report = classification_report(
            true_labels, predicted_labels, output_dict=True
        )

        return {
            "accuracy": accuracy,
            "confusion_matrix": conf_matrix,
            "classification_report": class_report,
        }

    def predict(self, image_path: str) -> List[Tuple[str, float]]:
        """Predict face identity in image."""
        if self.model is None:
            raise ValueError("Model not trained yet")

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.resize(image, (self.input_shape[0], self.input_shape[1]))
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)

        predictions = self.model.predict(image)
        top_indices = np.argsort(predictions[0])[::-1][:3]  # Top 3

        results = []
        for idx in top_indices:
            class_name = self.index_to_class.get(idx, "Unknown")
            confidence = float(predictions[0][idx])
            results.append((class_name, confidence))

        return results

    def save_model(self, path: str = "data/models/face_recognition.h5"):
        """Save trained model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load pre-trained model."""
        self.model = keras.models.load_model(path)
        print(f"Model loaded from {path}")


if __name__ == "__main__":
    recognizer = TensorFlowFaceRecognizer(num_classes=10)
    print("TensorFlow Face Recognizer initialized")
