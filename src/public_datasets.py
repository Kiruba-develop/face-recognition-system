"""Public Face Dataset Downloader and Preprocessor

Supported datasets:
- LFW (Labeled Faces in the Wild) - 13,233 images
- CelebA - 202,599 celebrity face images
- VGGFace2 - 3.31 million face images
- IMDB-WIKI - 500K+ face images from IMDb and Wikipedia
- UTKFace - 20,000+ face images
- MS-Celeb-1M - 10 million celebrity faces
- Faces94/Faces95/Faces96 - University of Essex dataset
"""

import os
import shutil
import urllib.request
import tarfile
import zipfile
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np


class PublicDatasetDownloader:
    """Download and process public face recognition datasets."""

    def __init__(self, output_dir="data/public_datasets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_lfw(self, download_pairs=False):
        """Download LFW (Labeled Faces in the Wild) dataset.
        
        Dataset: 13,233 images of 5,749 people
        Size: ~200 MB
        Source: http://vis-www.cs.umass.edu/lfw/
        """
        print("Downloading LFW dataset...")
        base_url = "http://vis-www.cs.umass.edu/lfw/"
        
        # Main dataset
        lfw_url = "lfw.tgz"
        download_path = self.output_dir / "lfw.tgz"
        
        print(f"Downloading from {base_url}{lfw_url}...")
        try:
            urllib.request.urlretrieve(
                base_url + lfw_url,
                download_path,
                reporthook=self._download_progress,
            )
            print(f"\nExtract LFW to {self.output_dir}/lfw")
            self._extract_tar(download_path, self.output_dir)
        except Exception as e:
            print(f"Error downloading LFW: {e}")
            print(f"\nManual download instructions:")
            print(f"1. Visit: {base_url}")
            print(f"2. Download lfw.tgz")
            print(f"3. Extract to {self.output_dir}/lfw")

        # Pairs file for verification task
        if download_pairs:
            pairs_url = "pairs.txt"
            print(f"\nDownloading {pairs_url}...")
            try:
                urllib.request.urlretrieve(
                    base_url + pairs_url,
                    self.output_dir / pairs_url,
                )
            except Exception as e:
                print(f"Error downloading pairs: {e}")

    def download_utk_face(self):
        """Download UTKFace dataset.
        
        Dataset: 20,000+ aligned and cropped face images
        Age, gender, ethnicity attributes included
        Source: https://susanqq.github.io/UTKFace/
        """
        print("Downloading UTKFace dataset...")
        base_url = "https://susanqq.github.io/UTKFace/"
        
        print(f"Manual download required for UTKFace.")
        print(f"1. Visit: {base_url}")
        print(f"2. Download the dataset")
        print(f"3. Extract to {self.output_dir}/utk_face")
        print(f"\nNote: UTKFace requires manual download due to server restrictions.")

    def download_celebrities_dataset(self):
        """Download simplified celebrity faces dataset.
        
        Alternative lightweight dataset for quick testing.
        """
        print("Downloading Celebrity Dataset (IMDB-WIKI subset)...")
        print(f"Note: For full VGGFace2/IMDB-WIKI, visit:")
        print(f"- VGGFace2: http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/")
        print(f"- IMDB-WIKI: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/")

    def download_faces94_95_96(self):
        """Download Faces94/95/96 datasets from University of Essex.
        
        Dataset: 3D face database
        Source: https://cswww.essex.ac.uk/mv/allfaces/
        """
        print("Downloading Faces94/95/96 dataset...")
        base_url = "https://cswww.essex.ac.uk/mv/allfaces/"
        print(f"Visit: {base_url}")
        print(f"Extract to: {self.output_dir}/faces94_95_96")

    def _download_progress(self, block_num, block_size, total_size):
        """Show download progress."""
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        print(f"\rDownloading: {percent:.1f}%", end="")

    def _extract_tar(self, tar_path, extract_to):
        """Extract tar file."""
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=extract_to)
        print(f"Extracted to {extract_to}")

    def _extract_zip(self, zip_path, extract_to):
        """Extract zip file."""
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extracted to {extract_to}")

    def organize_lfw_for_training(self, lfw_path=None, output_path=None):
        """Organize LFW dataset for training.
        
        LFW comes with one image per directory.
        Reorganize to have multiple images per person.
        """
        if lfw_path is None:
            lfw_path = self.output_dir / "lfw"
        if output_path is None:
            output_path = self.output_dir / "lfw_organized"

        output_path.mkdir(exist_ok=True)
        
        # Copy all images to person directories
        for person_dir in Path(lfw_path).iterdir():
            if person_dir.is_dir():
                person_name = person_dir.name
                output_person_dir = output_path / person_name
                output_person_dir.mkdir(exist_ok=True)
                
                for image_file in person_dir.glob("*.jpg"):
                    shutil.copy(image_file, output_person_dir / image_file.name)
        
        print(f"Organized LFW dataset saved to {output_path}")
        return output_path

    def preprocess_dataset(
        self,
        input_dir: str,
        output_dir: str,
        image_size: tuple = (224, 224),
        equalize_hist: bool = True,
        detect_faces: bool = True,
    ):
        """Preprocess face images (resize, normalize, detect faces)."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if detect_faces:
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

        for person_dir in input_path.iterdir():
            if not person_dir.is_dir():
                continue

            output_person_dir = output_path / person_dir.name
            output_person_dir.mkdir(exist_ok=True)

            for image_file in person_dir.glob("*.jpg"):
                try:
                    image = cv2.imread(str(image_file))
                    if image is None:
                        continue

                    # Detect faces if requested
                    if detect_faces:
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(gray)
                        if len(faces) == 0:
                            continue
                        x, y, w, h = faces[0]
                        image = image[y : y + h, x : x + w]

                    # Resize
                    image = cv2.resize(image, image_size)

                    # Equalize histogram
                    if equalize_hist:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        image = cv2.equalizeHist(image)
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

                    # Save
                    output_path_file = output_person_dir / image_file.name
                    cv2.imwrite(str(output_path_file), image)

                except Exception as e:
                    print(f"Error processing {image_file}: {e}")

        print(f"Preprocessing complete. Output saved to {output_path}")


class DatasetInfo:
    """Information about available public face datasets."""

    DATASETS = {
        "LFW": {
            "name": "Labeled Faces in the Wild",
            "size": "13,233 images",
            "people": "5,749 people",
            "url": "http://vis-www.cs.umass.edu/lfw/",
            "download_size": "~200 MB",
            "license": "Public Domain",
        },
        "CelebA": {
            "name": "CelebA Dataset",
            "size": "202,599 images",
            "people": "10,177 celebrity identities",
            "url": "http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html",
            "download_size": "~1.3 GB",
            "license": "Public (with citation required)",
        },
        "VGGFace2": {
            "name": "VGGFace2",
            "size": "3.31M images",
            "people": "9,131 identities",
            "url": "http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/",
            "download_size": "~~500 GB",
            "license": "Academic use only",
        },
        "UTKFace": {
            "name": "UTKFace",
            "size": "20,000+ images",
            "people": "Various",
            "url": "https://susanqq.github.io/UTKFace/",
            "download_size": "~1 GB",
            "license": "Academic/Research use",
            "attributes": "Age, Gender, Ethnicity",
        },
        "IMDB-WIKI": {
            "name": "IMDB-WIKI",
            "size": "500K+ images",
            "people": "Various celebrities",
            "url": "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/",
            "download_size": "~60 GB",
            "license": "Academic research use",
            "attributes": "Age, Gender, Year",
        },
        "FacesDB": {
            "name": "Faces94/95/96",
            "size": "Varies",
            "people": "152 identities",
            "url": "https://cswww.essex.ac.uk/mv/allfaces/",
            "download_size": "~100 MB",
            "license": "Free for research",
        },
        "MS-Celeb-1M": {
            "name": "MS-Celeb-1M",
            "size": "10M images",
            "people": "100K celebrities",
            "url": "https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-million-celebrities-in-the-wild/",
            "download_size": "~500 GB",
            "license": "Research only",
            "note": "Recommended: Use celebrity subsets only",
        },
    }

    @classmethod
    def print_info(cls):
        """Print info about all available datasets."""
        print("\n" + "="*80)
        print("PUBLIC FACE RECOGNITION DATASETS")
        print("="*80 + "\n")
        
        for key, info in cls.DATASETS.items():
            print(f"📊 {info['name']} ({key})")
            print(f"   Size: {info['size']}")
            print(f"   People: {info['people']}")
            print(f"   Download: {info['download_size']}")
            print(f"   License: {info['license']}")
            print(f"   URL: {info['url']}")
            if "attributes" in info:
                print(f"   Attributes: {info['attributes']}")
            if "note" in info:
                print(f"   Note: {info['note']}")
            print()


if __name__ == "__main__":
    DatasetInfo.print_info()
    
    # Example usage:
    # downloader = PublicDatasetDownloader()
    # downloader.download_lfw()
    # downloader.organize_lfw_for_training()
    # downloader.preprocess_dataset(
    #     input_dir="data/public_datasets/lfw_organized",
    #     output_dir="data/dataset/lfw_preprocessed"
    # )
