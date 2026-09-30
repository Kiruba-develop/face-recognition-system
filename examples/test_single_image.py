"""Test a single image against the trained face recognition model.

Usage:
    python examples/test_single_image.py path/to/image.jpg
    python examples/test_single_image.py path/to/image.jpg --model data/models/custom_face_model.h5 --database data/models/custom_identity_db.pkl --threshold 0.7
"""

import argparse
from pathlib import Path

from src.face_embedding_model import FaceEmbeddingModel


def main():
    parser = argparse.ArgumentParser(description="Recognize a face in a single image")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument(
        "--model",
        default="data/models/custom_face_model.h5",
        help="Path to trained embedding model",
    )
    parser.add_argument(
        "--database",
        default="data/models/custom_identity_db.pkl",
        help="Path to identity database",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.60,
        help="Recognition confidence threshold (default: 0.60)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of matches to display (default: 3)",
    )
    parser.add_argument(
        "--backbone",
        choices=["efficientnetb3", "mobilenetv2", "resnet50"],
        default="efficientnetb3",
        help="Backbone used to build the model",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    model_path = Path(args.model)
    database_path = Path(args.database)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not database_path.exists():
        raise FileNotFoundError(f"Identity database not found: {database_path}")

    model = FaceEmbeddingModel(backbone=args.backbone)
    model.load_models(str(model_path))
    model.load_identity_database(str(database_path))

    results = model.recognize_face(
        str(image_path),
        threshold=args.threshold,
        top_k=args.top_k,
    )

    print(f"Image: {image_path}")
    if not results or results[0][0] == "Unknown":
        print("Result: Unknown")
        return

    for name, confidence in results:
        print(f"{name}: confidence={confidence:.4f}")


if __name__ == "__main__":
    main()
