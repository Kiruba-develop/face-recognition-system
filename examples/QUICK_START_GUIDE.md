# ============================================================================
# DATASET SETUP + TESTING COMMANDS
# ============================================================================

"""
## Dataset folder creation

Linux/macOS:
  bash scripts/setup_dataset_folders.sh

Windows PowerShell:
  New-Item -ItemType Directory -Force `
    data/dataset/alice, `
    data/dataset/bob, `
    data/dataset/charlie, `
    data/known_faces/alice, `
    data/known_faces/bob, `
    data/known_faces/charlie, `
    data/models

Then copy your images:
  cp /path/to/alice/*.jpg data/dataset/alice/
  cp /path/to/bob/*.jpg data/dataset/bob/
  cp /path/to/charlie/*.jpg data/dataset/charlie/

## Single-image recognition test

python examples/test_single_image.py path/to/test-image.jpg

Example with custom thresholds:
  python examples/test_single_image.py path/to/test-image.jpg --threshold 0.70 --top-k 3

## Lower-resource training

python examples/low_resource_training.py

This runs a lightweight MobileNetV2 pipeline with:
  - epochs = 15
  - batch size = 4
  - lower GPU memory usage

## Webcam recognition

python examples/webcam_recognition.py \
  --model data/models/custom_face_model.h5 \
  --database data/models/custom_identity_db.pkl \
  --threshold 0.60
"""

print(__doc__)
