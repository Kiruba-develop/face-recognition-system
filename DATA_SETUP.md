# Data Setup Guide for Face Recognition System

This guide explains how to set up training datasets, known faces database, and pre-trained models for the face recognition system.

## Quick Start

### Option 1: Automatic Setup (Recommended)

```bash
# Check current data status
python setup_data.py --status

# Interactive setup (choose what to download)
python setup_data.py

# Auto setup all components
python setup_data.py --auto
```

### Option 2: Manual Setup

Create the required directory structure:

```bash
mkdir -p data/dataset
mkdir -p data/known_faces
mkdir -p data/models
```

Then manually populate directories with your data.

---

## Directory Structure

```
face-recognition-system/
├── data/
│   ├── dataset/                    # Training images (organized by person)
│   │   ├── person1/
│   │   │   ├── img1.jpg
│   │   │   ├── img2.jpg
│   │   │   └── ...
│   │   ├── person2/
│   │   │   └── ...
│   │   └── personN/
│   │
│   ├── known_faces/                # Pre-computed face encodings
│   │   ├── encodings.pkl           # Face embeddings database
│   │   └── names.pkl               # Corresponding person names
│   │
│   └── models/                     # Trained model files
│       ├── custom_model.h5         # Custom CNN model
│       └── transfer_learning.h5    # Transfer learning model
```

---

## Data Components

### 1. Training Dataset

**Purpose:** Train custom face recognition models

**Requirements:**
- Minimum **20-30 images per person** (100+ recommended)
- Image size: **80×80 pixels minimum** (preferably 224×224+)
- Clear face images with various angles and lighting
- Good quality, well-focused photographs
- Supported formats: JPG, PNG, BMP, TIFF

**Recommended Dataset Sources:**
- Your own collected images
- [LFW (Labeled Faces in the Wild)](http://vis-www.cs.umass.edu/lfw/)
- [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- [VGGFace2](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/)
- [MS-Celeb-1M](https://www.microsoft.com/en-us/research/project/ms-celeb-1m/)

**Setup:**

```bash
# Download your dataset
python setup_data.py --dataset

# Or manually organize in data/dataset/:
data/dataset/
├── john_doe/
│   ├── 001.jpg
│   ├── 002.jpg
│   └── ...
├── jane_smith/
│   ├── 001.jpg
│   └── ...
```

**Dataset Validation:**

```python
from src.utils import validate_dataset

issues = validate_dataset('data/dataset')
for issue in issues:
    print(issue)
```

---

### 2. Known Faces Database

**Purpose:** Store pre-computed face encodings for faster recognition

**Files:**
- `encodings.pkl` - 128-dimensional face embeddings
- `names.pkl` - Corresponding person names

**Creating the Database:**

```bash
# Method 1: Using the provided example
python examples/train_model.py --dataset data/dataset --build-database

# Method 2: Using Python API
python
>>> from src.database import FaceDatabase
>>> db = FaceDatabase()
>>> db.add_faces_from_directory('data/known_faces')
>>> db.save('data/known_faces/encodings.pkl')
```

**Pre-built Database:**

```bash
# Download pre-built database
python setup_data.py --known-faces
```

---

### 3. Pre-trained Models

**Purpose:** Use model weights for inference without training

**Supported Models:**
- `custom_model.h5` - Custom CNN (size: ~50-100MB)
- `transfer_learning.h5` - VGG/ResNet-based model (size: ~200MB)

**Getting Pre-trained Models:**

```bash
# Download pre-trained models
python setup_data.py --models

# Or download from releases
# https://github.com/Kiruba-develop/face-recognition-system/releases
```

**Model Specifications:**
- Format: TensorFlow/Keras `.h5`
- Input: 224×224×3 images
- Output: Class predictions or face embeddings
- Framework: TensorFlow 2.8+

---

## Data Source Configuration

### Setting Up Custom Data URLs

To enable automatic downloads, configure data sources in `setup_data.py`:

```python
DATA_SOURCES = {
    'dataset': {
        'url': 'https://your-storage.com/dataset.zip',
        'size': '~500MB',
    },
    'known_faces': {
        'url': 'https://your-storage.com/known-faces.zip',
        'size': '~50MB',
    },
    'models': {
        'url': 'https://your-storage.com/models.zip',
        'size': '~100MB',
    },
}
```

### Hosting Options

#### Google Drive

```bash
# 1. Upload file to Google Drive
# 2. Right-click → Share → Change to "Anyone with the link"
# 3. Get share link: https://drive.google.com/file/d/FILE_ID/view?usp=sharing
# 4. Convert to download URL:
https://drive.google.com/uc?export=download&id=FILE_ID
```

#### AWS S3

```bash
# Create public S3 bucket
aws s3 mb s3://my-face-recognition-data

# Upload data
aws s3 cp dataset.zip s3://my-face-recognition-data/

# Make public
aws s3api put-object-acl --bucket my-face-recognition-data \
  --key dataset.zip --acl public-read

# URL: https://my-face-recognition-data.s3.amazonaws.com/dataset.zip
```

#### GitHub Releases

```bash
# 1. Create a release in GitHub
# 2. Attach files (go to "Attach binaries by dropping them")
# 3. Copy download link: 
# https://github.com/owner/repo/releases/download/v1.0.0/dataset.zip
```

#### Hugging Face Datasets

```bash
# 1. Create account at https://huggingface.co
# 2. Create dataset repository
# 3. Upload files using Git LFS
# 4. Share URL
```

---

## Using Git LFS (Large File Storage)

For hosting large files directly in GitHub:

### Setup Git LFS

```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "data/dataset/**/*.jpg"
git lfs track "data/known_faces/**/*.pkl"
git lfs track "data/models/**/*.h5"

# Commit tracking configuration
git add .gitattributes
git commit -m "Configure Git LFS for large files"

# Upload files (Git LFS handles this automatically)
git add data/
git commit -m "Add training data and models"
git push
```

### Clone with LFS Files

```bash
# Clone repository (LFS files download automatically)
git clone https://github.com/Kiruba-develop/face-recognition-system.git

# Or initialize LFS for existing clone
git lfs install
git lfs pull
```

**Note:** GitHub's free tier includes 1GB LFS storage. Upgrade for more.

---

## Data Privacy & Security

### Best Practices

1. **Never commit unencrypted face encodings to public repos**
   ```bash
   # Add to .gitignore
   echo "data/known_faces/*.pkl" >> .gitignore
   ```

2. **Encrypt sensitive data**
   ```python
   from cryptography.fernet import Fernet
   
   # Encrypt encodings
   cipher = Fernet(key)
   encrypted = cipher.encrypt(encodings)
   ```

3. **Use environment variables for URLs**
   ```bash
   export DATA_SOURCE_URL="https://secure.example.com/data.zip"
   ```

4. **Implement access controls**
   - Use private GitHub repositories
   - Require authentication for data URLs
   - Enable branch protection

---

## Troubleshooting

### Problem: Files Not Downloading

```bash
# Check configuration guide
python setup_data.py --guide

# Verify URLs are configured
python setup_data.py --status

# Check network connection
curl https://your-storage.com/dataset.zip
```

### Problem: Corrupted Downloads

```bash
# Force re-download
python setup_data.py --force

# Or delete and retry
rm -rf data/dataset
python setup_data.py --dataset
```

### Problem: Insufficient Disk Space

```bash
# Check required space
du -sh data/

# Free up space or use external drive
# Configure alternative data path
```

### Problem: Git LFS Issues

```bash
# Check LFS status
git lfs status

# Reinstall LFS
git lfs install --force

# Fetch LFS files
git lfs pull --exclude=""
```

---

## Data Statistics

Example dataset statistics after setup:

```
Dataset Statistics:
  Total images: 5,000
  Total persons: 50
  Images per person: 100 ± 10
  Total size: 500 MB
  
  Training set: 70% (3,500 images)
  Validation set: 15% (750 images)
  Test set: 15% (750 images)
```

---

## Next Steps

After setting up data:

1. **Explore Dataset**
   ```bash
   jupyter notebook notebooks/explore_dataset.ipynb
   ```

2. **Train Custom Model**
   ```bash
   python examples/train_model.py --dataset data/dataset --epochs 50
   ```

3. **Test Recognition**
   ```bash
   python examples/basic_recognition.py
   ```

4. **Run Web Demo** (if available)
   ```bash
   python examples/webapp.py
   ```

---

## Support

For issues or questions:
- Check the [setup script help](setup_data.py)
- Review [main README.md](README.md)
- Open an [issue on GitHub](https://github.com/Kiruba-develop/face-recognition-system/issues)

---

**Last Updated:** 2026-09-30
