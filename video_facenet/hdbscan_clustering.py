from datetime import datetime
import umap
import joblib


import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)    
    import hdbscan

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

from .files import file_name


def cluster_faces_pipeline(reduce):
    models = []
    if reduce:
        models.append(
            ("umap", umap.UMAP(
                n_neighbors=30,
                min_dist=0.0,
                n_components=30,
                random_state=42,
            ))
        )
    models.append(("hdbscan", hdbscan.HDBSCAN (min_cluster_size=15)))
    return Pipeline(models)


class FaceCluster(BaseEstimator):
    def __init__(self, suffix, umap):
        super()
        self.suffix = suffix       
        self.clusterer = cluster_faces_pipeline(reduce=umap)

    @property
    def _file_name(self):
        return file_name("cluster{}.npy", self.suffix)

    def save(self):
        joblib.dump(self.clusterer, self._file_name)

    def load(self):
        fn = self._file_name
        if not os.path.isfile(fn):
            return False
        self.clusterer = joblib.load(fn)
        return True        

    @property
    def hdbscan(self)->umap.UMAP:
        return self.clusterer["hdbscan"]

    @property
    def umap(self):
        return self.clusterer["umap"]        

    @property
    def labels_(self):
        return self.hdbscan.labels_        

    def fit(self, X, y=None):
        return self.clusterer.fit(X, y)

    def fit_predict(self, X, y=None):
        start_time = datetime.now()        
        rt = self.clusterer.fit_predict(X, y)
        end_time = datetime.now()
        print("Clustering took: {}".format(end_time - start_time))        
        return rt

    def approximate_predict(self, X):
        return hdbscan.approximate_predict(self.hdbscan, X)

    def membership_vector(self, X):
        return hdbscan.membership_vector(self.hdbscan, X)

    @property
    def probabilities_(self):
        return self.hdbscan.probabilities_        

    def all_points_membership_vectors(self):
        return hdbscan.all_points_membership_vectors(self.hdbscan)

    def generate_prediction_data(self):
        self.hdbscan.generate_prediction_data()

    def set_params(self, **kwargs):
        return self.clusterer.set_params(**kwargs)