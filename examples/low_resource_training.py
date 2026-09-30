"""Low-resource training command helper.

This script runs a lightweight MobileNetV2 fine-tuning job to reduce GPU memory use.
Use it when your system cannot train the full EfficientNet pipeline.

Usage:
    python examples/low_resource_training.py
"""

from examples.fast_mobilenet_training import fast_mobilenet_training


if __name__ == "__main__":
    fast_mobilenet_training(
        dataset_path="data/dataset",
        known_faces_path="data/known_faces",
        model_save_path="data/models/fast_mobilenet_model.h5",
        database_save_path="data/models/fast_identity_db.pkl",
        epochs=15,
        batch_size=4,
    )
