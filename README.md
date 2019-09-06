## Facenet and face clustering

David Sandberg's [FaceNet](https://github.com/davidsandberg/facenet) and [MTCNN](https://github.com/ipazc/mtcnn) are packaged for easy video processing. Easily detect faces, generate Facenet embeddings and save extracted information  to *.csv* files. Tools for clustering and visualization based on HDBSCAN, umap, etc may help prepare data for additional Facenet training or for other supervised deep learning models. Or they could be used to tackle face identification tasks directly. Read [blog post](https://act-labs.github.io/posts/facenet-clustering/) for more details.  

## Installation
For ease of use, pip package is available as follows:
```
pip install video_facenet
```

Or just:
1. Clone repo
2. cd to base directory 
3. install requirements :
```
pip install -r requirements.txt
```


## Important Requirements
1. Python 3.5+ 
2. TensorFlow
3. OpenCV library

Some other packages were used, primarily for clustering and data visualization: pandas, scikit-learn, seaborn, etc. Could be useful, but may be redundant in some applications.

## Usage

To process video file:

```python
from video_facenet.pipelines import process_video
process_video(video_path=video_path, model_path=model_path, start=10, suffix="video_name", batch_size=64, end=None)

```

This function call could be used to process selected video file, starting at frame # 10. Bounding boxes, landmarks, and embeddings are saved into bounding_box_video_name.csv,landmarks_video_name.csv, and embeddings_video_name.csv files respectively. 

Module `video_facenet.cluster_analysis` contains functions for clustering and data visualization/analyses. The video.py file in the root repository provides example how to read parameters from *task.yaml* file and conduct clustering, visually explore extracted images, and save clustered images into subdirectories (with file paths conforming to *data/suffix/frames-or-images/cluster-id/probability_frame-id_closest-cluster0_closest-cluster1.jpg* scheme)

## Models and other resources

Original Facenet TensorFlow implementation as well as pre-trained models and datasets could be found in David Sandberg's [github repository](https://github.com/davidsandberg/facenet). Repository [Facial-Recognition-and-Alignment](https://github.com/armanrahman22/Facial-Recognition-and-Alignment) contains many utilities, models and links which were quite helpful. 







