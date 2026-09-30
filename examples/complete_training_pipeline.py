"""Complete end-to-end training pipeline

This script implements the full 3-stage training process:
1. Train on curated public datasets (LFW, CelebA, UTKFace, Faces94-96)
2. Fine-tune on custom identities
3. Build and save identity database for recognition
"""

import os
import logging
from pathlib import Path
from src.face_embedding_model import FaceEmbeddingModel
from src.public_datasets import PublicDatasetDownloader, DatasetInfo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def stage_1_prepare_public_datasets(output_dir: str = "data/unified_dataset_preprocessed"):
    """Stage 1: Download and prepare public datasets."""
    logger.info("\n" + "="*80)
    logger.info("STAGE 1: PREPARE PUBLIC DATASETS")
    logger.info("="*80)
    
    output_path = Path(output_dir)
    
    if output_path.exists() and len(list(output_path.glob("*"))) > 0:
        logger.info(f"Public datasets already prepared at {output_dir}")
        return output_dir
    
    logger.info("\nAvailable public datasets:")
    DatasetInfo.print_info()
    
    logger.info("\nTo download datasets:")
    logger.info("1. LFW (13K images, 200MB) - Automatically available")
    logger.info("   python -c \"from src.public_datasets import PublicDatasetDownloader; \"")
    logger.info("   d = PublicDatasetDownloader(); d.download_lfw()\"")
    logger.info()
    logger.info("2. CelebA (202K images, 1.3GB) - Manual download required")
    logger.info("   Visit: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html")
    logger.info()
    logger.info("3. UTKFace (20K images, 1GB) - Manual download")
    logger.info("   Visit: https://susanqq.github.io/UTKFace/")
    logger.info()
    logger.info("4. Faces94-96 (7K images, 100MB) - Manual download")
    logger.info("   Visit: https://cswww.essex.ac.uk/mv/allfaces/")
    
    return output_dir


def stage_2_train_on_public_data(
    model: FaceEmbeddingModel,
    public_dataset_path: str,
    epochs: int = 100,
    batch_size: int = 64,
):
    """Stage 2: Train embedding model on public datasets."""
    logger.info("\n" + "="*80)
    logger.info("STAGE 2: TRAIN ON PUBLIC DATASETS")
    logger.info("="*80)
    
    history = model.train_on_public_datasets(
        dataset_path=public_dataset_path,
        epochs=epochs,
        batch_size=batch_size,
        initial_learning_rate=0.001,
        checkpoint_path="data/models/public_dataset_checkpoint.h5",
    )
    
    return history


def stage_3_finetune_custom_identities(
    model: FaceEmbeddingModel,
    custom_dataset_path: str,
    epochs: int = 50,
    batch_size: int = 32,
):
    """Stage 3: Fine-tune on custom identities."""
    logger.info("\n" + "="*80)
    logger.info("STAGE 3: FINE-TUNE ON CUSTOM IDENTITIES")
    logger.info("="*80)
    
    if not Path(custom_dataset_path).exists():
        logger.warning(f"Custom dataset not found at {custom_dataset_path}")
        logger.warning("Skipping fine-tuning stage")
        return None
    
    history = model.fine_tune_on_custom_identities(
        dataset_path=custom_dataset_path,
        epochs=epochs,
        batch_size=batch_size,
        fine_tune_learning_rate=0.0001,
        freeze_backbone=False,
    )
    
    return history


def stage_4_build_identity_database(
    model: FaceEmbeddingModel,
    known_faces_dir: str,
    database_path: str = "data/models/identity_database.pkl",
):
    """Stage 4: Build identity database for recognition."""
    logger.info("\n" + "="*80)
    logger.info("STAGE 4: BUILD IDENTITY DATABASE")
    logger.info("="*80)
    
    if not Path(known_faces_dir).exists():
        logger.warning(f"Known faces directory not found at {known_faces_dir}")
        logger.warning("Skipping identity database creation")
        return
    
    model.build_identity_database(known_faces_dir)
    model.save_identity_database(database_path)
    
    logger.info(f"\nIdentity database saved to {database_path}")


def complete_training_pipeline(
    public_dataset_path: str = "data/unified_dataset_preprocessed",
    custom_dataset_path: str = "data/dataset",
    known_faces_dir: str = "data/known_faces",
    model_save_path: str = "data/models/face_embedding_model.h5",
    database_save_path: str = "data/models/identity_database.pkl",
    backbone: str = "efficientnetb3",
    public_epochs: int = 100,
    finetune_epochs: int = 50,
    batch_size: int = 64,
):
    """Execute complete 4-stage training pipeline.
    
    Args:
        public_dataset_path: Path to preprocessed public datasets
        custom_dataset_path: Path to custom identity training data
        known_faces_dir: Path to known faces for database
        model_save_path: Where to save embedding model
        database_save_path: Where to save identity database
        backbone: CNN backbone ('efficientnetb3', 'resnet50', 'mobilenetv2')
        public_epochs: Epochs for public dataset training
        finetune_epochs: Epochs for fine-tuning
        batch_size: Training batch size
    """
    logger.info("\n" + "#"*80)
    logger.info("# COMPLETE FACE EMBEDDING TRAINING PIPELINE")
    logger.info("#"*80)
    
    # Initialize model
    model = FaceEmbeddingModel(
        input_shape=(224, 224, 3),
        embedding_dim=128,
        backbone=backbone,
    )
    
    # Stage 1: Prepare public datasets
    logger.info("\nStage 1: Checking public datasets...")
    public_path = stage_1_prepare_public_datasets(public_dataset_path)
    
    # Stage 2: Train on public data
    logger.info("\nStage 2: Training on public datasets...")
    if Path(public_path).exists() and len(list(Path(public_path).glob("*"))) > 0:
        history_public = stage_2_train_on_public_data(
            model,
            public_path,
            epochs=public_epochs,
            batch_size=batch_size,
        )
        
        # Save model after public training
        model.save_models(
            embedding_path=model_save_path,
            classification_path=model_save_path.replace('.h5', '_classification.h5'),
        )
    else:
        logger.warning("Public dataset not available, skipping Stage 2")
        logger.info("You can still proceed with Stage 3 (fine-tuning on custom data)")
    
    # Stage 3: Fine-tune on custom identities
    logger.info("\nStage 3: Fine-tuning on custom identities...")
    history_finetune = stage_3_finetune_custom_identities(
        model,
        custom_dataset_path,
        epochs=finetune_epochs,
        batch_size=batch_size,
    )
    
    # Save model after fine-tuning
    model.save_models(
        embedding_path=model_save_path,
        classification_path=model_save_path.replace('.h5', '_classification.h5'),
    )
    
    # Stage 4: Build identity database
    logger.info("\nStage 4: Building identity database...")
    stage_4_build_identity_database(
        model,
        known_faces_dir,
        database_save_path,
    )
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("="*80)
    logger.info(f"\nModels saved to:")
    logger.info(f"  Embedding Model: {model_save_path}")
    logger.info(f"  Classification Model: {model_save_path.replace('.h5', '_classification.h5')}")
    logger.info(f"\nIdentity Database saved to:")
    logger.info(f"  {database_save_path}")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Load model: model.load_models('{model_save_path}')")
    logger.info(f"  2. Load database: model.load_identity_database('{database_save_path}')")
    logger.info(f"  3. Recognize: model.recognize_face('path/to/image.jpg')")
    
    return model


if __name__ == "__main__":
    # Run complete pipeline
    model = complete_training_pipeline(
        public_dataset_path="data/unified_dataset_preprocessed",
        custom_dataset_path="data/dataset",
        known_faces_dir="data/known_faces",
        model_save_path="data/models/face_embedding_model.h5",
        database_save_path="data/models/identity_database.pkl",
        backbone="efficientnetb3",
        public_epochs=100,
        finetune_epochs=50,
        batch_size=64,
    )
