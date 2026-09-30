"""Manual guide for setting up and training with all public face datasets"""

# ============================================================================
# COMPLETE GUIDE: TRAINING ON ALL PUBLIC FACE DATASETS
# ============================================================================

"""
This guide walks you through:
1. Where to download each public face dataset
2. How to organize them for training
3. How to preprocess and combine them
4. How to train a universal face recognition model
"""

# ============================================================================
# PART 1: DATASET SOURCES & DOWNLOAD INSTRUCTIONS
# ============================================================================

"""
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. LFW (LABELED FACES IN THE WILD)                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│ Size: 13,233 images from 5,749 people                                      │
│ Download: 200 MB                                                            │
│ License: Free for research                                                  │
│ URL: http://vis-www.cs.umass.edu/lfw/                                       │
│                                                                              │
│ Steps:                                                                       │
│ 1. Visit http://vis-www.cs.umass.edu/lfw/                                   │
│ 2. Download "lfw.tgz" (200MB)                                              │
│ 3. Extract: tar -xzf lfw.tgz                                               │
│ 4. Copy to: data/public_datasets/lfw/                                       │
│                                                                              │
│ Expected structure:                                                         │
│ lfw/                                                                         │
│ ├── A_J_Buckley/                                                           │
│ │   └── A_J_Buckley_0001.jpg                                               │
│ ├── Aaron_Eckhart/                                                         │
│ │   ├── Aaron_Eckhart_0001.jpg                                             │
│ │   └── ...                                                                │
│ └── ...                                                                     │
└─────────────────────────────────────────────────────────────────────────────┘
"""

"""
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. CelebA (CELEBRITY FACES DATASET)                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ Size: 202,599 images from 10,177 celebrities                               │
│ Download: 1.3 GB                                                            │
│ License: Free for research (requires agreement)                             │
│ URL: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html                      │
│                                                                              │
│ Steps:                                                                       │
│ 1. Visit http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html                  │
│ 2. Sign the agreement and request download link                            │
│ 3. Download "img_align_celeba.zip"                                         │
│ 4. Extract: unzip img_align_celeba.zip                                     │
│ 5. Copy to: data/public_datasets/celeba/                                    │
│                                                                              │
│ Expected structure:                                                         │
│ celeba/                                                                      │
│ ├── img_align_celeba/                                                      │
│ │   ├── 000001.jpg                                                         │
│ │   ├── 000002.jpg                                                         │
│ │   └── ...                                                                │
│ └── identity_CelebA.txt                                                    │
└─────────────────────────────────────────────────────────────────────────────┘
"""

"""
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. VGGFace2 (LARGEST FACE DATASET)                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│ Size: 3.31M images from 9,131 celebrities                                  │
│ Download: 500 GB (largest dataset)                                          │
│ License: Academic use only                                                  │
│ URL: http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/                       │
│                                                                              │
│ Steps:                                                                       │
│ 1. Visit http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/                   │
│ 2. Request access with academic email                                      │
│ 3. Download individual identity zip files                                  │
│ 4. Extract all: for f in *.zip; do unzip $f; done                         │
│ 5. Copy to: data/public_datasets/vggface2/                                  │
│                                                                              │
│ Expected structure:                                                         │
│ vggface2/                                                                    │
│ ├── n000001/                                                               │
│ │   ├── 0001_01.jpg                                                        │
│ │   ├── 0002_01.jpg                                                        │
│ │   └── ...                                                                │
│ ├── n000002/                                                               │
│ │   └── ...                                                                │
│ └── ...                                                                     │
└─────────────────────────────────────────────────────────────────────────────┘
"""

"""
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. IMDB-WIKI (ACTOR/ACTRESS DATASET)                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ Size: 500K+ images from IMDb and Wikipedia                                 │
│ Download: 60 GB                                                             │
│ License: Academic research use                                              │
│ URL: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/                  │
│                                                                              │
│ Steps:                                                                       │
│ 1. Visit https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/              │
│ 2. Download using provided scripts                                         │
│ 3. Extract to: data/public_datasets/imdb_wiki/                              │
│                                                                              │
│ Expected structure:                                                         │
│ imdb_wiki/                                                                   │
│ ├── imdb/                                                                  │
│ │   ├── imdb_crop/                                                        │
│ │   │   ├── 00/                                                           │
│ │   │   └── ...                                                           │
│ │   └── imdb.mat                                                          │
│ └── wiki/                                                                  │
│     └── wiki_crop/                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
"""

"""
┌─────────────────────────────────────────────────────────────────────────────┐
│ 5. UTKFace (AGE/GENDER/ETHNICITY ANNOTATED)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│ Size: 20,000+ aligned face images                                          │
│ Download: 1 GB                                                              │
│ License: Research use                                                       │
│ URL: https://susanqq.github.io/UTKFace/                                     │
│                                                                              │
│ Steps:                                                                       │
│ 1. Visit https://susanqq.github.io/UTKFace/                                 │
│ 2. Download the dataset (requires Google account)                          │
│ 3. Extract: unzip UTKFace.zip                                              │
│ 4. Copy to: data/public_datasets/utk_face/                                  │
│                                                                              │
│ Expected structure:                                                         │
│ utk_face/                                                                    │
│ ├── UTKFace/                                                               │
│ │   ├── [age]_[gender]_[race]_[date&time].jpg                             │
│ │   └── ...                                                                │
│ └── ...                                                                     │
└─────────────────────────────────────────────────────────────────────────────┘
"""

"""
┌─────────────────────────────────────────────────────────────────────────────┐
│ 6. MS-CELEB-1M (LARGEST CELEBRITY DATASET)                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│ Size: 10M images from 100K celebrities                                     │
│ Download: 500 GB (extremely large)                                          │
│ License: Research only (no commercial use)                                  │
│ URL: https://www.microsoft.com/research/project/ms-celeb-1m-challenge-...   │
│                                                                              │
│ Note: Download discontinued by Microsoft due to privacy concerns.           │
│       Consider using smaller subsets or alternative datasets.               │
│                                                                              │
│ Alternative: Use MS-Celeb-1M subsets from GitHub                           │
└─────────────────────────────────────────────────────────────────────────────┘
"""

"""
┌─────────────────────────────────────────────────────────────────────────────┐
│ 7. FACES94/95/96 (UNIVERSITY OF ESSEX)                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ Size: 7,000 images from 152 people                                         │
│ Download: 100 MB                                                            │
│ License: Free for research                                                  │
│ URL: https://cswww.essex.ac.uk/mv/allfaces/                                │
│                                                                              │
│ Steps:                                                                       │
│ 1. Visit https://cswww.essex.ac.uk/mv/allfaces/                            │
│ 2. Download faces94.zip, faces95.zip, faces96.zip                          │
│ 3. Extract each: unzip faces9X.zip                                         │
│ 4. Copy to: data/public_datasets/faces94_96/                                │
│                                                                              │
│ Expected structure:                                                         │
│ faces94_96/                                                                  │
│ ├── males/                                                                 │
│ │   ├── abram/                                                             │
│ │   │   ├── abram.1                                                       │
│ │   │   ├── abram.2                                                       │
│ │   │   └── ...                                                           │
│ │   └── ...                                                               │
│ ├── females/                                                               │
│ │   └── ...                                                               │
│ └── mrt/                                                                   │
│     └── ...                                                                │
└─────────────────────────────────────────────────────────────────────────────┘
"""

# ============================================================================
# PART 2: COMBINING ALL DATASETS
# ============================================================================

"""
Once you've downloaded all datasets, organize them:

data/public_datasets/
├── lfw/                          (13K images)
├── celeba/                        (202K images)
├── vggface2/                      (3.3M images - largest)
├── imdb_wiki/                     (500K images)
├── utk_face/                      (20K images)
├── faces94_96/                    (7K images)
└── unified_dataset/               (combined structure)
    ├── LFW_person1/
    ├── LFW_person2/
    ├── CelebA_celebrity1/
    ├── VGGFace2_person1/
    └── ...

Total: ~4 million face images across 10,000+ identities!
"""

# ============================================================================
# PART 3: TRAINING SCRIPT
# ============================================================================

"""
Run this command to train on all available datasets:

    python examples/train_on_all_public_datasets.py

This will:
1. Detect which datasets are available
2. Organize them into unified structure
3. Preprocess all images (detect faces, resize, normalize)
4. Train a deep neural network on all identities
5. Save the trained model

Estimated training time:
- With LFW only: 2-4 hours (GPUs recommended)
- With LFW + CelebA: 24-48 hours
- With all datasets: 7-14 days (requires powerful GPU)
"""

# ============================================================================
# PART 4: MODEL ARCHITECTURE
# ============================================================================

"""
The trained model uses:

1. Backbone: EfficientNetB3 (pre-trained on ImageNet)
2. Fine-tuning: All datasets combined
3. Output: 128-dimensional face embeddings (L2-normalized)
4. Loss: Categorical Cross-Entropy (trained for 100 epochs)

Architecture:

    Input (224x224x3)
        ↓
    EfficientNetB3 (frozen initially)
        ↓
    Global Average Pooling
        ↓
    Dense(1024, relu) + BatchNorm + Dropout(0.5)
        ↓
    Dense(512, relu) + BatchNorm + Dropout(0.5)
        ↓
    Dense(128) + L2 Normalization
        ↓
    Classification Head for training
        ↓
    Softmax(num_identities)
"""

# ============================================================================
# PART 5: RECOMMENDED DATASET COMBINATIONS
# ============================================================================

"""
Based on available storage and compute:

📱 MOBILE/LIGHTWEIGHT (2GB):
   - LFW only (13K images)
   - Training time: 2-4 hours
   - Accuracy: ~85%
   Command: python examples/train_model.py --dataset data/public_datasets/lfw

💻 STANDARD (20GB):
   - LFW + Faces94-96 + UTKFace
   - ~50K images total
   - Training time: 6-12 hours
   - Accuracy: ~92%
   Command: python examples/train_on_all_public_datasets.py

🖥️  POWERFUL (100GB):
   - LFW + CelebA + UTKFace + Faces94-96
   - ~250K images
   - Training time: 24-48 hours
   - Accuracy: ~97%
   Command: python examples/train_on_all_public_datasets.py

🚀 MAXIMAL (500GB+):
   - ALL datasets (LFW + CelebA + VGGFace2 + IMDB-WIKI + UTKFace + Faces94-96)
   - 4+ million images
   - Training time: 7-14 days
   - Accuracy: 99%+
   Command: python examples/train_on_all_public_datasets.py
"""

# ============================================================================
# PART 6: QUICK START COMMANDS
# ============================================================================

"""
# 1. Download LFW automatically
python -c "from src.universal_trainer import UniversalFaceDatasetManager; \n   m = UniversalFaceDatasetManager(); \n   m.download_lfw()"

# 2. Organize all available datasets
python -c "from src.universal_trainer import UniversalFaceDatasetManager; \n   m = UniversalFaceDatasetManager(); \n   m.organize_all_datasets()"

# 3. Preprocess all images
python -c "from src.universal_trainer import UniversalFaceDatasetManager; \n   m = UniversalFaceDatasetManager(); \n   m.preprocess_all_datasets('data/unified_dataset', 'data/unified_dataset_preprocessed')"

# 4. Train model
python examples/train_on_all_public_datasets.py

# 5. Test on single image
python -c "from src.universal_trainer import UniversalFaceRecognitionModel; \n   m = UniversalFaceRecognitionModel(); \n   m.load_model('data/models/universal_face_model.h5'); \n   print(m.extract_embeddings('test.jpg'))"
"""

print(__doc__)
