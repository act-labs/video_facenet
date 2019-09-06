from typing import Dict, Generator, List, Tuple
import numpy as np

Landmarks = Dict[str, Tuple[int, int]]
Embedding = np.ndarray
Image = np.ndarray
EmbeddingsGenerator = Generator[List[np.ndarray], None, None]

class AlignResult:
    def __init__(
            self, bounding_box: List[int], landmarks: Landmarks=None) -> None:
        # Bounding Box: [x1, y2, x2, y2]
        self.bounding_box = bounding_box
        self.landmarks = landmarks


class Face:
    """Class representing a single face

    Attributes:
        bounding_box {Float[]} -- box around their face in container_image
        image {Image} -- Image cropped around face
        embedding {Float} -- Face embedding
    """

    def __init__(self, bounding_box, landmarks):
        self.bounding_box = bounding_box
        self.landmarks = landmarks
        self.image = None
        self.embedding = None
        self.id = None
        self.pos = None

    def __str__ (self):
        return " id: {},\n bounding_box: {},\n landmarks:{}".format(self.id, self.bounding_box, self.landmarks)


FacesGenerator = Generator[List[Face], None, None]
ImageGenerator = Generator[np.ndarray, None, None]