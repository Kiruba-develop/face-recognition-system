"""Ready-to-run single script for training on custom dataset only

No public datasets required - trains directly on your own people.
Perfect for quick setup and personal face recognition.

Usage:
    python examples/quick_train_custom_only.py

Dataset structure:
    data/dataset/
    ├── alice/
    │   ├── alice_001.jpg
    │   ├── alice_002.jpg
    │   └── ...
    ├── bob/
    │   ├── bob_001.jpg
    │   └── ...
    └── charlie/
"""

import os
import logging
from pathlib import Path
from src.face_embedding_model import FaceEmbeddingModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def quick_train_custom_only(
    dataset_path: str = "data/dataset",
    known_faces_path: str = "data/known_faces",
    model_save_path: str = "data/models/custom_face_model.h5",
    database_save_path: str = "data/models/custom_identity_db.pkl",
    backbone: str = "efficientnetb3",
    epochs: int = 50,
    batch_size: int = 32,
):
    """Quick training on custom dataset only.
    
    Perfect for:
    - Personal face recognition
    - Small team/company
    - Quick setup without downloading public datasets
    - Testing before full training
    
    Requirements:
    - Minimum 20-30 images per person
    - Clear face images (80x80 minimum)
    - GPU recommended (4GB+)
    - Training time: 30 minutes - 2 hours
    """
    
    logger.info("\n" + "="*80)
    logger.info("QUICK CUSTOM-ONLY FACE RECOGNITION TRAINING")
    logger.info("="*80)
    
    dataset_path = Path(dataset_path)
    known_faces_path = Path(known_faces_path)
    
    # Check dataset exists
    if not dataset_path.exists():
        logger.error(f"Dataset not found at {dataset_path}")
        logger.error(f"\nPlease create dataset structure:")
        logger.error(f"  data/dataset/")
        logger.error(f"  ├── alice/")
        logger.error(f"  │   ├── alice_001.jpg")
        logger.error(f"  │   ├── alice_002.jpg")
        logger.error(f"  │   └── ...")
        logger.error(f"  ├── bob/")
        logger.error(f"  │   └── ...")
        return None
    
    # Count identities and images
    identities = [d for d in dataset_path.iterdir() if d.is_dir()]
    num_identities = len(identities)
    total_images = sum(len(list(d.glob("*.jpg"))) for d in identities)
    
    logger.info(f"\n📊 Dataset Summary:")
    logger.info(f"   Identities: {num_identities}")
    logger.info(f"   Total images: {total_images}")
    logger.info(f"   Average images per person: {total_images/max(num_identities, 1):.1f}")
    
    if num_identities < 2:
        logger.error("Need at least 2 identities for training")
        return None
    
    if total_images < 40:
        logger.warning(f"Recommended minimum 40 images total (have {total_images})")
        logger.warning("Model accuracy may be low with few images")
    
    # Initialize model
    logger.info(f"\n🏗️  Building model with {backbone} backbone...")
    model = FaceEmbeddingModel(
        input_shape=(224, 224, 3),
        embedding_dim=128,
        backbone=backbone,
    )
    
    # Train on custom data
    logger.info(f"\n🚀 Training on {num_identities} custom identities...")
    logger.info(f"   Epochs: {epochs}")
    logger.info(f"   Batch size: {batch_size}")
    logger.info(f"   Learning rate: 0.001")
    
    history = model.train_on_public_datasets(
        dataset_path=str(dataset_path),
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        initial_learning_rate=0.001,
        checkpoint_path=None,
    )
    
    # Save model
    logger.info(f"\n💾 Saving model to {model_save_path}...")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save_models(model_save_path)
    
    # Build identity database if known_faces exist
    if known_faces_path.exists() and len(list(known_faces_path.glob("*/"))) > 0:
        logger.info(f"\n🗄️  Building identity database from {known_faces_path}...")
        model.build_identity_database(str(known_faces_path))
        model.save_identity_database(database_save_path)
        logger.info(f"   Database saved to {database_save_path}")
        logger.info(f"   Identities in database: {len(model.identity_database)}")
    else:
        logger.warning(f"\nNo known_faces directory found at {known_faces_path}")
        logger.warning("Skipping identity database creation")
        logger.warning("You can add it later with: model.build_identity_database('path/to/faces')")
    
    # Print results
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"\n✅ Model saved: {model_save_path}")
    if known_faces_path.exists():
        logger.info(f"✅ Database saved: {database_save_path}")
    
    # Print final accuracy
    if history:
        final_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        logger.info(f"\n📈 Final Results:")
        logger.info(f"   Training accuracy: {final_acc:.2%}")
        logger.info(f"   Validation accuracy: {final_val_acc:.2%}")
    
    # Print usage instructions
    logger.info(f"\n📖 How to use the trained model:")
    logger.info(f"""
    from src.face_embedding_model import FaceEmbeddingModel
    
    # Load model
    model = FaceEmbeddingModel()
    model.load_models('{model_save_path}')
    model.load_identity_database('{database_save_path}')
    
    # Recognize a face
    results = model.recognize_face('image.jpg', threshold=0.6, top_k=3)
    for name, confidence in results:
        print(f'{{name}}: {{confidence:.2f}}')
    
    # Extract embedding
    embedding = model.extract_embedding('image.jpg')
    print(f'Embedding shape: {{embedding.shape}}')
    """)
    
    return model


if __name__ == "__main__":
    model = quick_train_custom_only(
        dataset_path="data/dataset",
        known_faces_path="data/known_faces",
        model_save_path="data/models/custom_face_model.h5",
        database_save_path="data/models/custom_identity_db.pkl",
        backbone="efficientnetb3",
        epochs=50,
        batch_size=32,
    )
