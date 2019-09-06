
import numpy as np
import os
from datetime import datetime
import shutil

import umap
import pandas as pd
import cv2

import matplotlib.pyplot as plt
import seaborn as sns

from .files import bounding_box_file_name, embeddings_file_name
from .hdbscan_clustering import FaceCluster
from .video import VideoProcessor

sns.set_context('poster')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.25, 's' : 80, 'linewidths':0}

def faces_slide_show(bounding_boxes, video, title="current", delay=0):
    if isinstance(video, str):
        video = VideoProcessor(video_path=video)
        print("opening video:\n%s" % video)

    for current in bounding_boxes.itertuples():
        video.pos = current.pos
        image = video.image
        cv2.rectangle(image, (current.x1, current.y1), (current.x2, current.y2),
                        (0, 255, 0), 2)    
        cv2.imshow(title, image)

        key = cv2.waitKey(delay)
        if key == ord('q') or key == ord('s'):
            return key

def open_bounding_boxes(suffix):
    return pd.read_csv(bounding_box_file_name(suffix), header=None, names=["pos", "id", "x1", "y1", "x2", "y2"])            

def show_found_faces(video_path, suffix="", delay=1):
    bounding_boxes = open_bounding_boxes(suffix)
    
    video = VideoProcessor(video_path=video_path)

    faces_slide_show(video=video, bounding_boxes=bounding_boxes, delay=delay)
    

def plot_clusters(labels, data):
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)    

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
        
    u = umap.UMAP(
        n_neighbors=10,
        min_dist=0.0,
        n_components=2,
        random_state=42,
    ).fit_transform(data)

    plt.scatter(u[:,0], u[:,1], c=colors, **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.title('{} clusters'.format(n_clusters_), fontsize=24)

    plt.show()   

def select_random_n(df, N):
    rndperm = np.random.RandomState(seed=42).permutation(df.shape[0])
    return df.iloc[rndperm[:N],:].copy()

def fit_or_open_model(suffix, values, clusterer, cache, umap, parameters):
    model = clusterer(suffix=suffix, umap=umap)
    if not cache or not model.load():
        if parameters:
            model.set_params(**parameters)
        model.fit_predict(values)
        model.save()
    return model

def add_cluster_assignments_info(df, model):
    model.generate_prediction_data()
    noise = model.labels_ == -1
    not_noise = ~noise        
    df["probabilities"] = model.probabilities_            
    all_points_membership_vectors = model.all_points_membership_vectors()    
    n, n_labels = all_points_membership_vectors.shape
    label_alternatives = np.zeros((n, 2), dtype=int)

    noise_membership = all_points_membership_vectors[noise]
    nonnoise_membership = all_points_membership_vectors[not_noise]
    label_alternatives[noise] = np.fliplr(np.argpartition(noise_membership, range(n_labels-2, n_labels))[:,-2:])
    label_alternatives[not_noise] = np.fliplr(np.argpartition(nonnoise_membership, range(n_labels-3, n_labels))[:,-3:-1])
    df["alt0"] = label_alternatives[:,0]
    df["alt1"] = label_alternatives[:,1]

def clean_images(suffix):
    path = "data/{}".format(suffix)
    if os.path.exists(path):
        shutil.rmtree(path)    

def dump_images(df, video, suffix, cropped=False):
    clusters = df.cluster.unique()

    directory = "images" if cropped else "frames"
    for cluster in clusters:
        os.makedirs("data/{}/{}/{}".format(suffix, directory, cluster), exist_ok=True)

    for current in df.itertuples():
        video.pos = current.pos
        image = video.image
        if cropped:
            image = image[current.y1:current.y2, current.x1:current.x2]
        else:
            cv2.rectangle(image, (current.x1, current.y1), (current.x2, current.y2), (0, 255, 0), 2)

        file_name = str(round(current.probabilities * 100)) + "_" + str(current.pos) + "_" + str(current.id) 
        file_name += "_" + str(current.alt0) + "_" + str(current.alt1)
        cv2.imwrite("data/{}/{}/{}/{}.jpg".format(suffix, directory, current.cluster, file_name), image)

  

def show_clusters(video_path, suffix, cache=False, parameters=None, delay=1, N=1000, umap=True, clusterer=FaceCluster, **kwargs):
    video = VideoProcessor(video_path=video_path)
    print("opening video:\n%s" % video)

    bounding_boxes = open_bounding_boxes(suffix)
    bounding_boxes["area"] = (bounding_boxes.x2 - bounding_boxes.x1) * (bounding_boxes.y2 - bounding_boxes.y1)
    bounding_boxes = bounding_boxes.sort_values("area", ascending=False)

    bounding_boxes = bounding_boxes.iloc[0:N]
    bounding_boxes = bounding_boxes.set_index(["pos", "id"], drop=False)


    embeddings = pd.read_csv(embeddings_file_name(suffix), header=None)
    columns = embeddings.columns
    embeddings.rename(columns={columns[0]:"pos", columns[1]:"id"}, inplace=True)
    embeddings = embeddings.set_index(["pos", "id"])
    print("embeddings:", embeddings.shape)

    embeddings = embeddings.loc[bounding_boxes.index]

    model = fit_or_open_model(suffix=suffix + "-" + str(N), values=embeddings.values, clusterer=clusterer, umap=umap, cache=cache, parameters=parameters)

    bounding_boxes["cluster"] = model.labels_
    add_cluster_assignments_info(df=bounding_boxes, model=model)

    clean_images(suffix=suffix)
    dump_images(df=bounding_boxes, video=video, suffix=suffix)
    dump_images(df=bounding_boxes, video=video, suffix=suffix, cropped=True)

    plot_clusters(labels=model.labels_, data=embeddings.values)

    clusters = bounding_boxes.cluster.value_counts()
    print("Cluster counts:", clusters)
    for id, total in clusters.items():
        key = faces_slide_show(video=video, bounding_boxes=bounding_boxes[bounding_boxes.cluster == id], delay=delay, title="cluster #{}, total {}".format(id, total))
        if key == ord("q"):
            return

    input("Enter any key to exit")