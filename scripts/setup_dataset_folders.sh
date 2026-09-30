#!/usr/bin/env bash
# Setup dataset directories for custom face recognition training.
# Run from repository root.

set -euo pipefail

mkdir -p data/dataset/{alice,bob,charlie}
mkdir -p data/known_faces/{alice,bob,charlie}
mkdir -p data/models

echo "Dataset folders created successfully."
echo ""
echo "Your structure is now:"
echo "data/"
echo "├── dataset/"
echo "│   ├── alice/"
echo "│   ├── bob/"
echo "│   └── charlie/"
echo "├── known_faces/"
echo "│   ├── alice/"
echo "│   ├── bob/"
echo "│   └── charlie/"
echo "└── models/"
echo ""
echo "Copy your images into the dataset folders manually:"
echo "  cp /path/to/alice/*.jpg data/dataset/alice/"
echo "  cp /path/to/bob/*.jpg data/dataset/bob/"
echo "  cp /path/to/charlie/*.jpg data/dataset/charlie/"
echo ""
echo "Add representative known face images to known_faces/ for the identity database."
