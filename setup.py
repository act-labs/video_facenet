from setuptools import find_packages, setup

setup(
    name='video_facenet',
    version='0.1.0',
    description="Face detection/embeding/clustering for video using TensorFlow",
    long_description="Face detection/embeddings/clustering for video  files using Google's FaceNet deep neural network & TensorFlow.",
    url='https://github.com/act-labs/video_facenet/',
    packages=find_packages(),
    maintainer='act7labs',
    maintainer_email='act7labs@gmail.com',
    include_package_data=True,
    license='MIT',
    install_requires=[
        'tensorflow',
        'scikit-learn',
        'pandas',        
        'opencv-python',
        'seaborn',
        'mtcnn',
        'umap-learn',
        'hdbscan'
    ])
