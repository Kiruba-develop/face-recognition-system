"""Quick reference guide for complete training pipeline"""

# ============================================================================
# COMPLETE FACE EMBEDDING TRAINING PIPELINE
# ============================================================================

"""
This guide walks through training a face embedding model using:
1. Modern CNN backbone (EfficientNetB3)
2. Curated public datasets (LFW, CelebA, UTKFace, Faces94-96)
3. Fine-tuning on custom identities
4. Identity database for recognition
"""

# ============================================================================
# QUICK START (5 MINUTES)
# ============================================================================

"""
# 1. Train on LFW only (smallest, fastest)
python -c "
from src.face_embedding_model import FaceEmbeddingModel
import logging
logging.basicConfig(level=logging.INFO)

model = FaceEmbeddingModel(backbone='efficientnetb3')
history = model.train_on_public_datasets(
    'data/unified_dataset_preprocessed',
    epochs=50,
    batch_size=64
)
model.save_models('data/models/model.h5')
"

# 2. Test recognition
python -c "
from src.face_embedding_model import FaceEmbeddingModel

model = FaceEmbeddingModel()
model.load_models('data/models/model.h5')
model.load_identity_database('data/models/identity_database.pkl')

results = model.recognize_face('test_image.jpg')
for name, confidence in results:
    print(f'{name}: {confidence:.2f}')
"
"""

# ============================================================================
# FULL TRAINING WORKFLOW
# ============================================================================

"""
## STEP 1: Prepare Datasets

dir structure:
  data/
  ├── unified_dataset_preprocessed/       <- Public datasets (preprocessed)
  │   ├── LFW_person1/
  │   ├── LFW_person2/
  │   ├── CelebA_celebrity1/
  │   └── ...
  ├── dataset/                           <- Custom identities (for fine-tuning)
  │   ├── alice/
  │   │   ├── alice_001.jpg
  │   │   ├── alice_002.jpg
  │   │   └── ...
  │   ├── bob/
  │   │   ├── bob_001.jpg
  │   │   └── ...
  │   └── charlie/
  ├── known_faces/                        <- Known faces (for database)
  │   ├── alice/
  │   │   ├── alice_profile.jpg
  │   │   └── ...
  │   ├── bob/
  │   └── ...
  └── models/                            <- Output models
      ├── face_embedding_model.h5
      ├── identity_database.pkl
      └── ...

## STEP 2: Download Public Datasets

# LFW (automatic)
from src.public_datasets import PublicDatasetDownloader
d = PublicDatasetDownloader()
d.download_lfw()

# CelebA, UTKFace, Faces94-96 - Manual download from links in TRAIN_ON_PUBLIC_DATASETS.md

## STEP 3: Organize and Preprocess

from src.universal_trainer import UniversalFaceDatasetManager
m = UniversalFaceDatasetManager()
m.organize_all_datasets('data/unified_dataset')
m.preprocess_all_datasets('data/unified_dataset', 'data/unified_dataset_preprocessed')

## STEP 4: Train Complete Pipeline

python examples/complete_training_pipeline.py

OR use programmatically:

from examples.complete_training_pipeline import complete_training_pipeline

model = complete_training_pipeline(
    public_dataset_path='data/unified_dataset_preprocessed',
    custom_dataset_path='data/dataset',
    known_faces_dir='data/known_faces',
    backbone='efficientnetb3',
    public_epochs=100,
    finetune_epochs=50,
    batch_size=64
)

"""

# ============================================================================
# TRAINING STAGES EXPLAINED
# ============================================================================

"""
## STAGE 1: PREPARE PUBLIC DATASETS

Inputs: Downloaded public face datasets (LFW, CelebA, UTKFace, etc.)
Outputs: Organized, preprocessed dataset structure

What happens:
  1. Detect faces in all images
  2. Crop face region
  3. Align to 224x224
  4. Histogram equalization
  5. Organize by identity

Time: 1-2 hours (depending on dataset size)

---

## STAGE 2: TRAIN ON PUBLIC DATASETS

Inputs: Preprocessed public datasets (~4M images, 100K identities)
Outputs: Trained embedding model + backbone

What happens:
  1. Initialize EfficientNetB3 backbone (pretrained on ImageNet)
  2. Add embedding layers (1024 -> 512 -> 128 dimensions)
  3. Add L2 normalization
  4. Add classification head for training
  5. Train with:
     - Adam optimizer (LR: 0.001)
     - Categorical crossentropy loss
     - Data augmentation (rotation, zoom, shift, flip)
     - Early stopping + LR scheduling
     - Validation split: 20%

Time: 24-168 hours (1-7 days, depending on data size and GPU)
BatchSize: 64 (larger = faster, but need more memory)
Epochs: 100 (stops early if validation loss plateaus)

Expected accuracy: 95-99% on validation set

---

## STAGE 3: FINE-TUNE ON CUSTOM IDENTITIES

Inputs: Trained model + custom identity dataset
Outputs: Fine-tuned model for your specific people

What happens:
  1. Unfreeze backbone for fine-tuning
  2. Lower learning rate (0.0001)
  3. Train only on custom identities
  4. Use less aggressive augmentation
  5. Smaller batch size (32) for stability

Time: 1-6 hours
Epochs: 50
Minimum images per person: 20 images

Why fine-tuning helps:
  - Public datasets have celebrities, diverse people
  - Your people have specific lighting, angles, expressions
  - Fine-tuning adapts the model to your faces
  - Usually improves accuracy by 5-10% on custom data

---

## STAGE 4: BUILD IDENTITY DATABASE

Inputs: Fine-tuned model + known faces directory
Outputs: Identity database (pickle file)

What happens:
  1. For each known person:
     a. Extract embedding from all their images
     b. Compute mean embedding
     c. Store in database
  2. Save database to disk

Database structure:
  {
    'alice': {
      'embedding': [array of 128 floats],
      'num_samples': 15
    },
    'bob': {
      'embedding': [...],
      'num_samples': 10
    },
    ...
  }

Time: Few minutes

"""

# ============================================================================
# INFERENCE/RECOGNITION
# ============================================================================

"""
## Using Trained Model

# Load everything
from src.face_embedding_model import FaceEmbeddingModel

model = FaceEmbeddingModel()
model.load_models('data/models/face_embedding_model.h5')
model.load_identity_database('data/models/identity_database.pkl')

# Recognize face
results = model.recognize_face(
    'image.jpg',
    threshold=0.6,      # Confidence threshold (0-1)
    top_k=3             # Return top 3 matches
)

for name, confidence in results:
    print(f'{name}: {confidence:.2f}')

# Extract embedding
embedding = model.extract_embedding('image.jpg')  # Returns 128D vector

# Compare two faces
embed1 = model.extract_embedding('face1.jpg')
embed2 = model.extract_embedding('face2.jpg')
similarity = np.dot(embed1, embed2)  # Cosine similarity
print(f'Similarity: {similarity:.2f}')  # High = same person

"""

# ============================================================================
# ARCHITECTURE DETAILS
# ============================================================================

"""
## Model Architecture

Embedding Model:
  Input (224x224x3)
    ↓
  EfficientNetB3 backbone (trained on ImageNet)
    ↓ (frozen during stage 2, unfrozen during stage 3)
  GlobalAveragePooling2D
    ↓
  Dense(1024, relu) + BatchNorm + Dropout(0.5)
    ↓
  Dense(512, relu) + BatchNorm + Dropout(0.5)
    ↓
  Dense(128)  # Embedding layer
    ↓
  L2 Normalization  # Force unit norm
    ↓
  Output: 128-dimensional face embedding

Classification Model (for training):
  Embedding Model output (128D)
    ↓
  Dense(num_classes, softmax)  # One per identity
    ↓
  Loss: Categorical Crossentropy
    ↓
  During inference: Removed, only use embeddings

## Why this architecture?

1. EfficientNetB3 backbone:
   - Pretrained on ImageNet (millions of images)
   - Efficient (fewer parameters than ResNet)
   - Provides strong feature extraction
   - Transfer learning accelerates convergence

2. Embedding layer:
   - 128 dimensions (standard for face recognition)
   - Captures identity-specific features
   - Smaller than 1024D (more efficient)
   - Larger than 64D (more expressive)

3. L2 normalization:
   - Forces embeddings onto unit sphere
   - Makes cosine similarity equivalent to dot product
   - Improves numerical stability
   - Better for ArcFace-style losses

4. Dropout & BatchNorm:
   - Regularization prevents overfitting
   - Batch norm accelerates training
   - Crucial when fine-tuning on small datasets

"""

# ============================================================================
# TIPS & BEST PRACTICES
# ============================================================================

"""
## For Best Results:

1. DATA QUALITY
   - Use clear, frontal face images (80x80 minimum)
   - Avoid extreme angles or heavy occlusion
   - Good lighting helps
   - 20-50 images per person (diminishing returns after 50)

2. TRAINING
   - Use batch size 64 for public data (good GPU memory balance)
   - Use batch size 32 for fine-tuning (smaller dataset)
   - Monitor validation loss (should decrease steadily)
   - If loss plateaus, reduce learning rate or stop early

3. HYPERPARAMETERS
   - Public training LR: 0.001 (default)
   - Fine-tune LR: 0.0001 (10x smaller)
   - Early stopping: patience=15 (public), patience=10 (fine-tune)
   - Dropout: 0.5 (strong regularization)

4. FINE-TUNING
   - Start with public-trained model (don't train from scratch)
   - Unfreezing backbone helps if you have enough custom data (100+ images)
   - Keep backbone frozen if you have <100 custom images
   - Use lower learning rate (0.0001, not 0.001)

5. RECOGNITION
   - Threshold 0.6-0.7 is good default (adjust based on false positive rate)
   - If model is too permissive (false positives), increase threshold
   - If model is too strict (false negatives), decrease threshold
   - Always normalize embeddings (built-in with L2 norm)

6. HARDWARE
   - GPU strongly recommended (10-100x faster than CPU)
   - 16GB+ GPU RAM needed for batch_size=64
   - CPU training: 10-20x slower
   - Can use smaller batch size on limited GPU memory

7. DEBUGGING
   - If accuracy stuck low: Check data quality
   - If overfitting: Increase augmentation or dropout
   - If underfitting: Decrease regularization, increase epochs
   - If recognition fails: Lower threshold or add more known faces

"""

# ============================================================================
# COMPARISON: Different Configurations
# ============================================================================

"""
Config 1: LIGHTWEIGHT (Fast, Low Memory)
  - Backbone: MobileNetV2
  - Public dataset: LFW only
  - Batch size: 32
  - Training time: 2-4 hours
  - Memory: 4GB GPU
  - Accuracy: ~85%
  
Config 2: BALANCED (Recommended)
  - Backbone: EfficientNetB3
  - Public dataset: LFW + CelebA + UTKFace + Faces94-96
  - Batch size: 64
  - Training time: 24-48 hours
  - Memory: 12GB GPU
  - Accuracy: ~97%
  
Config 3: POWERFUL (Best Accuracy)
  - Backbone: EfficientNetB3
  - Public dataset: ALL (LFW + CelebA + VGGFace2 + IMDB-WIKI + UTKFace + Faces94-96)
  - Batch size: 64
  - Training time: 7-14 days
  - Memory: 16GB+ GPU
  - Accuracy: 99%+
"""

print(__doc__)
