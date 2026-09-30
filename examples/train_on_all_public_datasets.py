"""Quick start guide for training on public datasets"""

import logging
from pathlib import Path
from src.universal_trainer import UniversalFaceDatasetManager, UniversalFaceRecognitionModel
from src.public_datasets import PublicDatasetDownloader, DatasetInfo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_and_train_full_pipeline():
    """
    Complete pipeline:
    1. Download public datasets
    2. Organize and preprocess
    3. Train universal model
    """
    
    logger.info("="*80)
    logger.info("UNIVERSAL FACE RECOGNITION TRAINING PIPELINE")
    logger.info("="*80)
    
    # Step 1: Show available datasets
    logger.info("\nStep 1: Available Public Datasets")
    logger.info("-"*80)
    DatasetInfo.print_info()
    
    # Step 2: Dataset setup
    logger.info("\nStep 2: Setting up datasets...")
    logger.info("-"*80)
    dataset_manager = UniversalFaceDatasetManager(
        root_dir="data/public_datasets"
    )
    
    # Download automatically available dataset (LFW)
    logger.info("Attempting to download LFW (automatically available)...")
    if dataset_manager.download_lfw():
        logger.info("✓ LFW downloaded successfully")
    else:
        logger.warning("✗ LFW download failed. You can download manually from:")
        logger.warning("  http://vis-www.cs.umass.edu/lfw/")
    
    # Instructions for other datasets
    logger.info("\nStep 3: Download Manual Datasets")
    logger.info("-"*80)
    logger.info("Download these datasets and extract to data/public_datasets/:")
    logger.info()
    logger.info("CelebA (202K images):")
    logger.info("  Download: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html")
    logger.info("  Extract to: data/public_datasets/celeba")
    logger.info()
    logger.info("VGGFace2 (3.3M images):")
    logger.info("  Download: http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/")
    logger.info("  Extract to: data/public_datasets/vggface2")
    logger.info()
    logger.info("IMDB-WIKI (500K+ images):")
    logger.info("  Download: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/")
    logger.info("  Extract to: data/public_datasets/imdb_wiki")
    logger.info()
    logger.info("UTKFace (20K images):")
    logger.info("  Download: https://susanqq.github.io/UTKFace/")
    logger.info("  Extract to: data/public_datasets/utk_face")
    logger.info()
    
    # Step 4: Organize datasets
    logger.info("\nStep 4: Organizing datasets...")
    logger.info("-"*80)
    unified_dataset = dataset_manager.organize_all_datasets(
        output_dir="data/unified_dataset"
    )
    
    # Step 5: Preprocess
    logger.info("\nStep 5: Preprocessing images...")
    logger.info("-"*80)
    logger.info("This step will:")
    logger.info("  - Detect faces in each image")
    logger.info("  - Crop to face region")
    logger.info("  - Resize to 224x224")
    logger.info("  - Equalize histogram")
    logger.info("  - Save preprocessed versions")
    logger.info()
    
    dataset_manager.preprocess_all_datasets(
        input_dir=str(unified_dataset),
        output_dir="data/unified_dataset_preprocessed",
        image_size=(224, 224),
    )
    
    # Step 6: Train model
    logger.info("\nStep 6: Training Universal Model")
    logger.info("-"*80)
    logger.info("Starting training on all public datasets...")
    logger.info()
    
    model = UniversalFaceRecognitionModel(
        input_shape=(224, 224, 3),
        embedding_size=128,
    )
    
    history = model.train_on_public_datasets(
        dataset_path="data/unified_dataset_preprocessed",
        epochs=100,
        batch_size=64,
        validation_split=0.2,
        backbone="efficientnetb3",
        model_save_path="data/models/universal_face_model.h5",
    )
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*80)
    logger.info(f"Model saved to: data/models/universal_face_model.h5")
    logger.info(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    logger.info(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    
    return model, history


if __name__ == "__main__":
    model, history = setup_and_train_full_pipeline()
