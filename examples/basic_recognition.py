import os
import cv2


def list_images(directory):
    image_paths = []
    for root, _, files in os.walk(directory):
        for file_name in files:
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_paths.append(os.path.join(root, file_name))
    return image_paths


def resize_image(image, width=224, height=224):
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def ensure_directory(path):
    os.makedirs(path, exist_ok=True)
    return path


def format_confidence(value):
    return max(0, min(1, float(value)))


if __name__ == '__main__':
    print('Utilities loaded successfully.')

