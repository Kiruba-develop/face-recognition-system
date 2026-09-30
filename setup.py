from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='face-recognition-system',
    version='1.0.0',
    author='Kiruba-develop',
    author_email='your-email@example.com',
    description='A comprehensive face recognition system using OpenCV, TensorFlow, and face_recognition libraries',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Kiruba-develop/face-recognition-system',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Multimedia :: Graphics :: Graphics Conversion',
    ],
    python_requires='>=3.7',
    install_requires=[
        'opencv-python>=4.5.0',
        'face-recognition>=1.3.0',
        'tensorflow>=2.0.0',
        'keras>=2.0.0',
        'numpy>=1.19.0',
        'scikit-learn>=0.24.0',
        'pillow>=8.0.0',
        'matplotlib>=3.3.0',
    ],
)
