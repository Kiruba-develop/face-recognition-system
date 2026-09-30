# Face Recognition System

A comprehensive face recognition system using OpenCV, TensorFlow, and face_recognition libraries with custom model training capabilities.

## Features

✅ **Face Detection & Recognition**
- Real-time face detection using OpenCV
- Fast face recognition with face_recognition library
- Support for webcam and image file inputs

✅ **Custom Model Training**
- Train your own deep learning model with TensorFlow/Keras
- Support for custom face datasets
- Model evaluation and performance metrics

✅ **Face Database Management**
- Store and manage face encodings
- Add/remove faces from database
- Support for multiple identities

✅ **Real-time Recognition**
- Live webcam face recognition
- Real-time display of detected faces and identities
- Confidence scores for each match

✅ **Multi-method Approach**
- Pre-trained model recognition (face_recognition library)
- Custom trained CNN model
- Ensemble methods for better accuracy

## Project Structure

```
face-recognition-system/
├── README.md
├── requirements.txt
├── setup.py
├── data/
│   ├── dataset/              # Custom training dataset
│   │   ├── person1/
│   │   ├── person2/
│   │   └── ...
│   ├── known_faces/          # Database of known face encodings
│   └── models/               # Trained models
├── src/
│   ├── __init__.py
│   ├── face_detector.py      # Face detection module
│   ├── face_recognizer.py    # Face recognition module
│   ├── model_trainer.py      # Custom model training
│   ├── database.py           # Face database management
│   └── utils.py              # Utility functions
├── notebooks/
│   ├── explore_dataset.ipynb
│   ├── train_custom_model.ipynb
│   └── evaluate_model.ipynb
├── examples/
│   ├── basic_recognition.py
│   ├── webcam_recognition.py
│   ├── batch_recognition.py
│   └── train_model.py
└── tests/
    └── test_recognition.py
```

## Installation

### Prerequisites
- Python 3.7+
- pip or conda

### Quick Start

```bash
# Clone the repository
git clone https://github.com/Kiruba-develop/face-recognition-system.git
cd face-recognition-system

# Install dependencies
pip install -r requirements.txt

# Or using conda
conda create -n face-recognition python=3.8
conda activate face-recognition
pip install -r requirements.txt
```

## Usage

### 1. Basic Face Recognition (Pre-trained Model)

```python
from src.face_recognizer import FaceRecognizer

# Initialize recognizer
recognizer = FaceRecognizer()

# Recognize faces in image
results = recognizer.recognize_image('path/to/image.jpg')

for person, confidence in results:
    print(f"Detected: {person} (Confidence: {confidence:.2f})")
```

### 2. Real-time Webcam Recognition

```bash
python examples/webcam_recognition.py
```

### 3. Train Custom Model

```bash
python examples/train_model.py --dataset data/dataset --epochs 50 --batch-size 32
```

### 4. Build Face Database

```python
from src.database import FaceDatabase

db = FaceDatabase()
db.add_faces_from_directory('data/known_faces')
db.save('data/known_faces/encodings.pkl')
```

## Architecture & Implementation

### Method 1: Pre-trained Model (face_recognition library)

**Architecture:**
- Uses dlib's CNN-based face detector
- ResNet-based face encoding model
- Outputs 128-dimensional face embeddings
- Compares embeddings using Euclidean distance

**Pros:**
- Fast and accurate
- Pre-trained on large dataset
- Easy to implement

**Cons:**
- Limited customization
- Fixed embedding size

### Method 2: Custom CNN Model (TensorFlow/Keras)

**Architecture:**
```
Input Image (224x224x3)
  ↓
Conv2D (32, 3x3) + ReLU → MaxPool
  ↓
Conv2D (64, 3x3) + ReLU → MaxPool
  ↓
Conv2D (128, 3x3) + ReLU → MaxPool
  ↓
Flattening
  ↓
Dense (256) + ReLU + Dropout(0.5)
  ↓
Dense (128) + ReLU
  ↓
Dense (num_classes) + Softmax
```

**Training:**
- Loss: Categorical Crossentropy (Softmax) or Triplet Loss
- Optimizer: Adam
- Metrics: Accuracy, Precision, Recall

**Pros:**
- Customizable architecture
- Train on your own data
- Better performance on specific faces

**Cons:**
- Requires more training data
- Longer training time
- GPU recommended

### Method 3: Transfer Learning (VGG-Face or ResNet)

**Architecture:**
- Use pre-trained backbone (VGG, ResNet)
- Remove classification layers
- Add custom classification head
- Fine-tune on your dataset

**Pros:**
- Combines speed and customization
- Better accuracy with less data
- Faster training

**Cons:**
- Still requires training data
- More complex implementation

## Step-by-Step Training Guide

### Step 1: Prepare Dataset
```
data/dataset/
├── person1/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── person2/
│   ├── img1.jpg
│   └── ...
└── personN/
    └── ...
```

**Dataset Requirements:**
- Minimum 20-30 images per person
- Clear face images (80x80 minimum)
- Various angles and lighting conditions
- Good quality images

### Step 2: Train Model

```python
from src.model_trainer import ModelTrainer

trainer = ModelTrainer()
history = trainer.train(
    dataset_path='data/dataset',
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    model_type='custom'  # or 'transfer_learning'
)

trainer.save_model('data/models/custom_model.h5')
```

### Step 3: Evaluate Model

```python
from src.model_trainer import ModelTrainer

trainer = ModelTrainer()
metrics = trainer.evaluate('data/dataset/test', 'data/models/custom_model.h5')

print(f"Accuracy: {metrics['accuracy']:.2%}")
print(f"Precision: {metrics['precision']:.2%}")
print(f"Recall: {metrics['recall']:.2%}")
```

## Performance Benchmarks

| Method | Speed | Accuracy | Training Time | Customization |
|--------|-------|----------|---------------|---------------|
| face_recognition | ⚡⚡⚡ Fast | 99%+ | N/A | ❌ Low |
| Custom CNN | ⚡⚡ Medium | 95-98% | High | ✅ High |
| Transfer Learning | ⚡⚡⚡ Fast | 97-99% | Medium | ✅ High |

## Advanced Features

### 1. Face Verification
```python
# Verify if two images are same person
is_same = recognizer.verify_face('image1.jpg', 'image2.jpg')
```

### 2. Face Detection Only
```python
# Detect faces without recognition
faces = detector.detect_faces('image.jpg')
for (x, y, w, h) in faces:
    print(f"Face at ({x}, {y}) with size {w}x{h}")
```

### 3. Batch Processing
```python
# Process multiple images
results = recognizer.recognize_batch('path/to/images/')
```

### 4. Export Encodings
```python
# Save face encodings for faster recognition
db.export_encodings('encodings.pkl')
```

## Troubleshooting

### Issue: Low accuracy
- Add more training images per person
- Improve image quality (lighting, focus)
- Increase number of epochs
- Try transfer learning
- Adjust confidence threshold

### Issue: Slow recognition
- Reduce image resolution
- Use GPU acceleration
- Use batch processing
- Cache encodings

### Issue: Face not detected
- Ensure face is clearly visible
- Check image brightness
- Increase detection scale
- Use higher resolution image

## Dependencies

- **OpenCV** - Face detection
- **face_recognition** - Pre-trained face recognition
- **TensorFlow/Keras** - Custom model training
- **NumPy** - Numerical operations
- **Pillow** - Image processing
- **scikit-learn** - Machine learning utilities
- **matplotlib** - Visualization

See `requirements.txt` for complete list.

## Best Practices

1. **Dataset Quality** - Use diverse, high-quality images
2. **Data Augmentation** - Apply rotation, brightness, zoom variations
3. **Regular Updates** - Retrain model periodically with new data
4. **Privacy** - Implement privacy-preserving face storage (encrypted encodings)
5. **Testing** - Test on unseen data before deployment
6. **Monitoring** - Track accuracy and false positives in production
7. **Threshold Tuning** - Adjust confidence threshold based on use case

## Examples

See `examples/` directory for:
- Basic recognition
- Real-time webcam recognition
- Batch processing
- Model training
- Database management

## License

MIT License

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## References

- [face_recognition Library](https://github.com/ageitgey/face_recognition)
- [OpenCV Face Detection](https://docs.opencv.org/master/)
- [TensorFlow Face Recognition](https://www.tensorflow.org/tutorials/generative/dcgan)
- [FaceNet Paper](https://arxiv.org/abs/1503.03832)
- [DeepFace Paper](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf)

## Support

For issues and questions, please open an issue on GitHub.
