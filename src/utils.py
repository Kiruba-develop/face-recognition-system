import os
from typing import Dict, List

import cv2
import numpy as np
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class ModelTrainer:
    """Custom CNN trainer for face recognition."""

    def __init__(self, input_shape=(224, 224, 3), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None

    def build_model(self):
        model = models.Sequential([
            layers.Input(shape=self.input_shape),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax'),
        ])
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )
        self.model = model
        return model

    def train(self, dataset_path, epochs=20, batch_size=32, validation_split=0.2):
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f'Dataset not found: {dataset_path}')

        train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
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

        self.num_classes = train_generator.num_classes
        self.build_model()

        history = self.model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            steps_per_epoch=train_generator.samples // batch_size,
            validation_steps=validation_generator.samples // batch_size,
        )

        return history

    def evaluate(self, test_dataset_path):
        if self.model is None:
            raise ValueError('Model has not been trained yet.')

        test_datagen = ImageDataGenerator(rescale=1./255)
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
        report = classification_report(true_labels, predicted_labels, output_dict=True)
        matrix = confusion_matrix(true_labels, predicted_labels)

        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': matrix,
        }

    def save_model(self, path='data/models/custom_model.h5'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)

    def load_model(self, path):
        self.model = models.load_model(path)
        return self.model
