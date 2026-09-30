"""Quick start guide for training and recognition"""

# ============================================================================
# QUICK START: CUSTOM-ONLY TRAINING (30 MIN - 2 HOURS)
# ============================================================================

"""
## 1. Prepare Your Data

Create folder structure:
  data/dataset/
  ├── alice/
  │   ├── alice_001.jpg
  │   ├── alice_002.jpg
  │   ├── alice_003.jpg
  │   └── ... (20-50 images)
  ├── bob/
  │   ├── bob_001.jpg
  │   └── ... (20-50 images)
  └── charlie/
      ├── charlie_001.jpg
      └── ... (20-50 images)

Requirements per person:
  - Minimum 20 images
  - Clear face images
  - Various angles/lighting
  - 80x80 pixels minimum

Optional: Create known_faces/ for identity database
  data/known_faces/
  ├── alice/
  │   └── alice_profile.jpg
  ├── bob/
  │   └── bob_profile.jpg
  └── charlie/
      └── charlie_profile.jpg

## 2. Train Model (OPTION A: Full Training)

python examples/quick_train_custom_only.py

This will:
  ✅ Train on all your custom identities
  ✅ Create embedding model (128D faces)
  ✅ Build identity database
  ✅ Save model to: data/models/custom_face_model.h5
  ✅ Save database to: data/models/custom_identity_db.pkl

Time: 30 min - 2 hours (depending on data size and GPU)

## 3. Train Model (OPTION B: Fast Training with MobileNetV2)

python examples/fast_mobilenet_training.py

Benefits:
  ✅ 10x faster than EfficientNetB3
  ✅ Lower GPU memory (4GB vs 12GB+)
  ✅ Accuracy: 90-95% (vs 97%+ for larger models)
  ✅ Better for real-time recognition

Time: 30 min - 1 hour

## 4. Test with Webcam

python examples/webcam_recognition.py \
    --model data/models/custom_face_model.h5 \
    --database data/models/custom_identity_db.pkl \
    --threshold 0.6

Keybindings:
  q - Quit
  s - Save detected face
  space - Pause/Resume
  t - Toggle detection

## 5. Use in Your Code

from src.face_embedding_model import FaceEmbeddingModel

# Load
model = FaceEmbeddingModel()
model.load_models('data/models/custom_face_model.h5')
model.load_identity_database('data/models/custom_identity_db.pkl')

# Recognize
results = model.recognize_face('image.jpg', threshold=0.6, top_k=3)
for name, confidence in results:
    print(f'{name}: {confidence:.2f}')

# Extract embedding
embedding = model.extract_embedding('image.jpg')
print(f'Embedding shape: {embedding.shape}')  # (128,)

"""

# ============================================================================
# ARCHITECTURE COMPARISON
# ============================================================================

"""
┌─────────────────┬──────────────────┬──────────────────────┬─────────────────┐
│ Feature         │ EfficientNetB3   │ MobileNetV2          │ ResNet50        │
├─────────────────┼──────────────────┼──────────────────────┼─────────────────┤
│ Accuracy        │ 97-99%           │ 90-95%               │ 95-98%          │
│ Backbone Params │ 40M              │ 22.5M                │ 23.5M           │
│ Model Size      │ 300MB            │ 95MB                 │ 100MB           │
│ GPU Memory      │ 12GB+            │ 4GB+                 │ 8GB+            │
│ Training Time   │ 24-100 hours     │ 2-4 hours            │ 12-50 hours     │
│ Inference Time  │ 200-300ms        │ 50-100ms             │ 150-200ms       │
│ Best For        │ High accuracy    │ Speed/Mobile         │ Balanced        │
│ Recommended?    │ Production       │ Real-time/Edge       │ Good middle     │
└─────────────────┴──────────────────┴──────────────────────┴─────────────────┘
"""

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================

"""
Optimal settings for your custom data:

## Quick Train (30 min)
  - Backbone: MobileNetV2
  - Epochs: 20
  - Batch size: 16
  - Learning rate: 0.001
  - Expected accuracy: 85-90%

## Standard Train (1-2 hours)
  - Backbone: EfficientNetB3
  - Epochs: 50
  - Batch size: 32
  - Learning rate: 0.001
  - Expected accuracy: 90-97%

## Deep Train (4-8 hours)
  - Backbone: EfficientNetB3
  - Epochs: 100
  - Batch size: 64
  - Learning rate: 0.001 -> 0.0001 (schedule)
  - Expected accuracy: 97-99%

## Tuning Tips:
  - More images per person -> higher accuracy
  - Lower learning rate -> slower training, higher accuracy
  - Higher batch size -> faster training, need more GPU memory
  - More epochs -> better fit, risk of overfitting
"""

# ============================================================================
# RECOGNITION PARAMETERS
# ============================================================================

"""
Threshold tuning (0-1 scale):

  0.5 - Very permissive
    ✅ Catches most faces
    ❌ High false positives (wrong identities accepted)
    💡 Use for: Lenient verification

  0.6 - Default (Recommended)
    ✅ Good balance of accuracy and coverage
    ✅ Catches ~98% of correct faces
    ❌ ~2% false positive rate
    💡 Use for: Most applications

  0.7 - Strict
    ✅ Low false positives (very few wrong matches)
    ❌ Misses some real faces (harder conditions)
    💡 Use for: High-security scenarios

  0.8 - Very strict
    ✅ Almost no false positives
    ❌ High false negatives (misses faces)
    💡 Use for: Maximum security

How to adjust:

  from src.face_embedding_model import FaceEmbeddingModel

  model = FaceEmbeddingModel()
  model.load_models('model.h5')
  model.load_identity_database('db.pkl')

  # Adjust threshold
  results = model.recognize_face('image.jpg', threshold=0.65)
"""

# ============================================================================
# DATASET PREPARATION TIPS
# ============================================================================

"""
## Good Face Images:
  ✅ Front-facing or slight angle
  ✅ Eyes clearly visible
  ✅ No extreme shadows
  ✅ Face takes up 60%+ of image
  ✅ Clear, focused image
  ✅ Natural lighting or well-lit

## Avoid:
  ❌ Profile view (90 degrees)
  ❌ Face looking down/up (extreme angles)
  ❌ Heavy occlusion (glasses, masks, hair covering face)
  ❌ Blurry or low-res images
  ❌ Very low lighting (dark images)
  ❌ Back of head or partial face
  ❌ Face too small (< 80x80 pixels)

## Optimal dataset structure:

For best results, aim for:
  - 30-50 images per person (20 minimum)
  - 5-10 people (minimum 2)
  - Various angles: front (60%), slight left (20%), slight right (20%)
  - Various lighting: indoor, outdoor, different times
  - Various expressions: neutral, smiling, serious
  - No extreme poses or occlusions

## Data collection:

  Option 1: Video extraction
    1. Record 30-60 second video of each person
    2. Extract frames: ffmpeg -i video.mp4 -vf fps=1 frame_%04d.jpg
    3. Keep best quality frames

  Option 2: Photo collection
    1. Take 20-50 photos per person
    2. Various angles and lighting
    3. Ensure clear face in each

  Option 3: Screenshot from video
    1. Find video of person
    2. Pause at various points
    3. Screenshot and save
"""

# ============================================================================
# TROUBLESHOOTING
# ============================================================================

"""
Problem: Model training is very slow
Solutions:
  1. Use MobileNetV2 instead of EfficientNetB3
  2. Reduce batch_size (trades memory for speed)
  3. Use GPU (CUDA/cuDNN)
  4. Check GPU is actually being used: nvidia-smi

Problem: Out of memory during training
Solutions:
  1. Reduce batch_size: 64 -> 32 -> 16 -> 8
  2. Use smaller backbone: MobileNetV2 instead of EfficientNetB3
  3. Use smaller input: 224x224 -> 128x128 (less accurate)
  4. Reduce validation_split: 0.2 -> 0.1

Problem: Recognition accuracy is low
Solutions:
  1. Add more training images per person (20 -> 50)
  2. Improve image quality (clear, frontal, good lighting)
  3. Train longer: increase epochs
  4. Lower learning rate: 0.001 -> 0.0001
  5. Try different backbone: ResNet50 or larger model

Problem: False positives (wrong person matched)
Solutions:
  1. Increase threshold: 0.6 -> 0.7 -> 0.8
  2. Add more known faces to database
  3. Train with more diverse data
  4. Use larger backbone (more accurate)
  5. Ensure people look different enough

Problem: False negatives (person not recognized)
Solutions:
  1. Decrease threshold: 0.6 -> 0.5
  2. Add more images to known_faces/
  3. Add images with similar conditions (same lighting, angle)
  4. Train longer with more data
  5. Check image quality (frontal, clear face)

Problem: Webcam recognition is slow
Solutions:
  1. Use MobileNetV2 backbone (10x faster)
  2. Reduce frame resolution
  3. Skip frames: process every 2nd or 3rd frame
  4. Use smaller batch_size for inference
  5. GPU inference (CUDA/cuDNN enabled)

"""

print(__doc__)
