"""Advanced Face Embedding Model with Modern CNN Backbone

Implements:
- ArcFace-style embeddings for better face separation
- EfficientNetB3/ResNet50 backbone with transfer learning
- Triplet loss for metric learning
- Complete training pipeline on public datasets
- Fine-tuning on custom identities
- Custom identity database for recognition
"""

import os
import cv2
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3, ResNet50, MobileNetV2
from sklearn.metrics import accuracy_score, roc_auc_score
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArcFaceLoss(keras.losses.Loss):
    """ArcFace loss for face recognition.
    
    Creates an angular margin between different identities.
    """
    
    def __init__(self, num_classes, margin=0.5, scale=64):
        super().__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
    
    def call(self, y_true, y_pred):
        # y_pred shape: (batch_size, embedding_size)
        # y_true shape: (batch_size, num_classes)
        
        # Normalize embeddings
        embeddings = tf.nn.l2_normalize(y_pred, axis=1)
        
        # Get class weights
        weights = tf.Variable(tf.random.normal((self.num_classes, tf.shape(y_pred)[1])))
        weights = tf.nn.l2_normalize(weights, axis=1)
        
        # Compute logits
        logits = tf.matmul(embeddings, tf.transpose(weights)) * self.scale
        
        # Add angular margin
        theta = tf.math.acos(logits / self.scale)
        theta_margin = theta + self.margin
        
        # Apply margin
        logits_margin = tf.cos(theta_margin) * self.scale
        
        # Compute loss
        loss = keras.losses.categorical_crossentropy(y_true, logits_margin, from_logits=True)
        return tf.reduce_mean(loss)


class TripletLoss(keras.losses.Loss):
    """Triplet loss for face recognition.
    
    Minimizes distance between positive pair and maximizes with negative pair.
    """
    
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    
    def call(self, y_true, y_pred):
        # This is typically implemented with data sampling strategy
        # Simplified version here
        anchor = y_pred[:, :128]
        positive = y_pred[:, 128:256]
        negative = y_pred[:, 256:]
        
        # L2 distance
        pos_distance = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        neg_distance = tf.reduce_sum(tf.square(anchor - negative), axis=1)
        
        # Triplet loss
        loss = tf.maximum(pos_distance - neg_distance + self.margin, 0.0)
        return tf.reduce_mean(loss)


class FaceEmbeddingModel:
    """Face embedding model using modern CNN backbone.
    
    Features:
    - Pre-trained backbone (EfficientNetB3/ResNet50)
    - 128-dimensional L2-normalized embeddings
    - ArcFace-style training
    - Multi-stage training strategy
    - Custom identity fine-tuning
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        embedding_dim: int = 128,
        backbone: str = "efficientnetb3",
    ):
        self.input_shape = input_shape
        self.embedding_dim = embedding_dim
        self.backbone_name = backbone
        
        self.embedding_model = None
        self.classification_model = None
        self.class_indices = {}
        self.index_to_class = {}
        self.identity_database = {}
        
        logger.info(f"Initialized FaceEmbeddingModel with {backbone} backbone")
    
    def _build_backbone(self) -> keras.Model:
        """Build and return pre-trained backbone model."""
        if self.backbone_name == "efficientnetb3":
            backbone = EfficientNetB3(
                weights="imagenet",
                include_top=False,
                input_shape=self.input_shape,
            )
        elif self.backbone_name == "resnet50":
            backbone = ResNet50(
                weights="imagenet",
                include_top=False,
                input_shape=self.input_shape,
            )
        elif self.backbone_name == "mobilenetv2":
            backbone = MobileNetV2(
                weights="imagenet",
                include_top=False,
                input_shape=self.input_shape,
            )
        else:
            raise ValueError(f"Unknown backbone: {self.backbone_name}")
        
        logger.info(f"Loaded {self.backbone_name} backbone")
        return backbone
    
    def build_embedding_model(self, freeze_backbone: bool = True) -> keras.Model:
        """Build embedding model.
        
        Architecture:
            Input -> Backbone -> GlobalAvgPool -> Dense layers -> 
            L2 Normalization -> Embedding
        """
        backbone = self._build_backbone()
        
        if freeze_backbone:
            backbone.trainable = False
            logger.info("Backbone weights frozen")
        
        model = models.Sequential([
            layers.Input(shape=self.input_shape),
            backbone,
            layers.GlobalAveragePooling2D(),
            layers.Dense(1024, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.embedding_dim),
            layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1)),  # L2 normalization
        ], name="embedding_model")
        
        self.embedding_model = model
        logger.info(f"Built embedding model with {self.embedding_dim}D output")
        return model
    
    def build_classification_model(self, num_classes: int) -> keras.Model:
        """Build classification model for training.
        
        Combines embedding model with classification head.
        """
        if self.embedding_model is None:
            self.build_embedding_model()
        
        embedding_input = layers.Input(shape=self.input_shape)
        embedding = self.embedding_model(embedding_input)
        
        # Classification head with ArcFace-style output
        output = layers.Dense(
            num_classes,
            activation='softmax',
            kernel_regularizer=keras.regularizers.l2(1e-4),
            name='classification_head',
        )(embedding)
        
        model = keras.Model(inputs=embedding_input, outputs=output, name="classification_model")
        self.classification_model = model
        logger.info(f"Built classification model for {num_classes} identities")
        return model
    
    def compile_model(
        self,
        learning_rate: float = 0.001,
        loss: str = "categorical_crossentropy",
    ):
        """Compile the classification model."""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.classification_model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[
                'accuracy',
                keras.metrics.Precision(),
                keras.metrics.Recall(),
                keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy'),
            ],
        )
        logger.info(f"Compiled model with {loss} loss")
    
    def train_on_public_datasets(
        self,
        dataset_path: str,
        epochs: int = 100,
        batch_size: int = 64,
        validation_split: float = 0.2,
        initial_learning_rate: float = 0.001,
        checkpoint_path: Optional[str] = None,
    ) -> keras.callbacks.History:
        """Train on curated public datasets.
        
        Stage 1: Train on diverse public dataset mix
        """
        logger.info("="*80)
        logger.info("STAGE 1: TRAINING ON PUBLIC DATASETS")
        logger.info("="*80)
        
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        # Data generators with aggressive augmentation for public data
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=40,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.2,
            zoom_range=0.3,
            horizontal_flip=True,
            fill_mode='reflect',
            validation_split=validation_split,
        )
        
        train_gen = train_datagen.flow_from_directory(
            dataset_path,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            seed=42,
        )
        
        val_gen = train_datagen.flow_from_directory(
            dataset_path,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            seed=42,
        )
        
        num_classes = train_gen.num_classes
        self.class_indices = train_gen.class_indices
        self.index_to_class = {v: k for k, v in self.class_indices.items()}
        
        logger.info(f"Training on {num_classes} identities")
        logger.info(f"Training samples: {train_gen.samples}")
        logger.info(f"Validation samples: {val_gen.samples}")
        
        # Build models
        self.build_embedding_model(freeze_backbone=True)
        self.build_classification_model(num_classes)
        self.compile_model(learning_rate=initial_learning_rate)
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
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
        ]
        
        if checkpoint_path:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            callbacks.append(
                keras.callbacks.ModelCheckpoint(
                    checkpoint_path,
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1,
                )
            )
        
        # Train
        history = self.classification_model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )
        
        logger.info("Public dataset training complete")
        return history
    
    def fine_tune_on_custom_identities(
        self,
        dataset_path: str,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2,
        fine_tune_learning_rate: float = 0.0001,
        freeze_backbone: bool = True,
    ) -> keras.callbacks.History:
        """Fine-tune on custom identities.
        
        Stage 2: Fine-tune on your own people for personalized recognition
        """
        logger.info("="*80)
        logger.info("STAGE 2: FINE-TUNING ON CUSTOM IDENTITIES")
        logger.info("="*80)
        
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        # Less aggressive augmentation for fine-tuning
        finetune_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            validation_split=validation_split,
        )
        
        train_gen = finetune_datagen.flow_from_directory(
            dataset_path,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            seed=42,
        )
        
        val_gen = finetune_datagen.flow_from_directory(
            dataset_path,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            seed=42,
        )
        
        # Update classification model for new classes
        num_custom_classes = train_gen.num_classes
        self.build_classification_model(num_custom_classes)
        
        # Unfreeze and fine-tune backbone
        if not freeze_backbone and self.embedding_model.layers[1].trainable == False:
            self.embedding_model.layers[1].trainable = True
            logger.info("Unfroze backbone for fine-tuning")
        
        self.compile_model(learning_rate=fine_tune_learning_rate)
        
        logger.info(f"Fine-tuning on {num_custom_classes} custom identities")
        logger.info(f"Training samples: {train_gen.samples}")
        logger.info(f"Validation samples: {val_gen.samples}")
        
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
                patience=3,
                min_lr=1e-10,
                verbose=1,
            ),
        ]
        
        # Fine-tune
        history = self.classification_model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )
        
        logger.info("Custom identity fine-tuning complete")
        return history
    
    def extract_embedding(self, image_path: str) -> np.ndarray:
        """Extract 128D embedding from image."""
        if self.embedding_model is None:
            raise ValueError("Embedding model not built")
        
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Detect and crop face
        face = self._detect_and_crop_face(image)
        if face is None:
            raise ValueError(f"No face detected in {image_path}")
        
        # Preprocess
        face = cv2.resize(face, (self.input_shape[0], self.input_shape[1]))
        face = face.astype('float32') / 255.0
        face = np.expand_dims(face, axis=0)
        
        # Extract embedding
        embedding = self.embedding_model.predict(face, verbose=0)
        return embedding[0]
    
    def _detect_and_crop_face(self, image: np.ndarray, scale: float = 1.1) -> Optional[np.ndarray]:
        """Detect and crop face from image."""
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale,
            minNeighbors=5,
            minSize=(30, 30),
        )
        
        if len(faces) == 0:
            return None
        
        # Get largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face = image[y:y+h, x:x+w]
        return face
    
    def build_identity_database(self, identities_dir: str) -> Dict[str, np.ndarray]:
        """Build database of known identity embeddings.
        
        Stage 3: Create embedding database for recognition
        """
        logger.info("="*80)
        logger.info("STAGE 3: BUILDING IDENTITY DATABASE")
        logger.info("="*80)
        
        identities_dir = Path(identities_dir)
        self.identity_database = {}
        
        for person_dir in sorted(identities_dir.iterdir()):
            if not person_dir.is_dir():
                continue
            
            person_name = person_dir.name
            embeddings = []
            
            logger.info(f"Processing {person_name}...")
            
            for image_file in person_dir.glob("*.jpg"):
                try:
                    embedding = self.extract_embedding(str(image_file))
                    embeddings.append(embedding)
                except Exception as e:
                    logger.warning(f"Failed to process {image_file}: {e}")
            
            if embeddings:
                # Store mean embedding
                mean_embedding = np.mean(embeddings, axis=0)
                self.identity_database[person_name] = {
                    'embedding': mean_embedding,
                    'num_samples': len(embeddings),
                    'all_embeddings': embeddings,
                }
                logger.info(f"  Added {person_name} with {len(embeddings)} samples")
        
        logger.info(f"\nIdentity database built with {len(self.identity_database)} identities")
        return self.identity_database
    
    def recognize_face(
        self,
        image_path: str,
        threshold: float = 0.6,
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """Recognize face in image.
        
        Returns top-k matches with similarity scores.
        """
        if not self.identity_database:
            raise ValueError("Identity database is empty")
        
        try:
            embedding = self.extract_embedding(image_path)
        except Exception as e:
            logger.error(f"Failed to extract embedding: {e}")
            return []
        
        # Compute similarities
        similarities = []
        for name, data in self.identity_database.items():
            # Cosine similarity
            similarity = np.dot(embedding, data['embedding']) / (
                np.linalg.norm(embedding) * np.linalg.norm(data['embedding']) + 1e-6
            )
            similarities.append((name, float(similarity)))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Filter by threshold
        results = [(name, sim) for name, sim in similarities[:top_k] if sim >= threshold]
        
        if not results:
            return [("Unknown", 0.0)]
        
        return results
    
    def save_identity_database(self, path: str):
        """Save identity database to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Convert numpy arrays to lists for pickling
        db_to_save = {}
        for name, data in self.identity_database.items():
            db_to_save[name] = {
                'embedding': data['embedding'].tolist(),
                'num_samples': data['num_samples'],
            }
        
        with open(path, 'wb') as f:
            pickle.dump(db_to_save, f)
        
        logger.info(f"Identity database saved to {path}")
    
    def load_identity_database(self, path: str):
        """Load identity database from disk."""
        with open(path, 'rb') as f:
            db_loaded = pickle.load(f)
        
        self.identity_database = {}
        for name, data in db_loaded.items():
            self.identity_database[name] = {
                'embedding': np.array(data['embedding']),
                'num_samples': data['num_samples'],
            }
        
        logger.info(f"Loaded identity database with {len(self.identity_database)} identities")
    
    def save_models(self, embedding_path: str, classification_path: Optional[str] = None):
        """Save trained models."""
        os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
        
        self.embedding_model.save(embedding_path)
        logger.info(f"Embedding model saved to {embedding_path}")
        
        if classification_path and self.classification_model:
            self.classification_model.save(classification_path)
            logger.info(f"Classification model saved to {classification_path}")
    
    def load_models(self, embedding_path: str, classification_path: Optional[str] = None):
        """Load trained models."""
        self.embedding_model = keras.models.load_model(embedding_path)
        logger.info(f"Embedding model loaded from {embedding_path}")
        
        if classification_path and os.path.exists(classification_path):
            self.classification_model = keras.models.load_model(classification_path)
            logger.info(f"Classification model loaded from {classification_path}")


if __name__ == "__main__":
    logger.info("Face Embedding Model initialized")
