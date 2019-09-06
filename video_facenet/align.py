import os

import warnings

from typing import List, cast

import cv2

import numpy as np

from mtcnn.mtcnn import MTCNN

from .utils import fix_mtcnn_bb, preprocess, get_center_box
from .facenet_types import AlignResult, Face, Landmarks, Image


warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Detector:
    def __init__(
            self,
            face_crop_height: int = 160,
            face_crop_width: int = 160,
            face_crop_margin: float = .4,
            min_face_size: int = 20,

            is_rgb: bool = True,

            use_affine: bool = False) -> None:

        self.mtcnn = MTCNN()
        self.face_crop_height = face_crop_height
        self.face_crop_width = face_crop_width
        self.face_crop_margin = face_crop_margin
        self.min_face_size = min_face_size
        self.is_rgb = is_rgb
        self.use_affine = use_affine

    def find_faces(self, image: Image, detect_multiple_faces: bool = True) -> List[Face]:
        faces = []
        results = cast(List[AlignResult], self._get_align_results(image))

        if not detect_multiple_faces and len(results) > 1:
            img_size = np.asarray(image.shape)[0:2]
            results = get_center_box(img_size, results)
        for result in results:
            face = Face(bounding_box=result.bounding_box, landmarks=result.landmarks)
            bb = result.bounding_box
            if bb[2] - bb[0] < self.min_face_size or bb[3] - \
                    bb[1] < self.min_face_size:
                pass
            # preprocess changes RGB -> BGR
            processed = preprocess(
                image,
                self.face_crop_height,
                self.face_crop_width,
                self.face_crop_margin,
                bb,
                result.landmarks,
                self.use_affine)
            resized = cv2.resize(
                processed, (self.face_crop_height, self.face_crop_width))
            # RGB to BGR
            if not self.is_rgb:
                resized = resized[..., ::-1]
            face.image = resized
            faces.append(face)
        return faces

    def _get_align_results(self, image: Image) -> List[AlignResult]:
        mtcnn_results = self.mtcnn.detect_faces(image)
        img_size = np.asarray(image.shape)[0:2]
        align_results = cast(List[AlignResult], [])

        for result in mtcnn_results:
            bb = result['box']
            # bb[x, y, dx, dy] -> bb[x1, y1, x2, y2]
            bb = fix_mtcnn_bb(
                img_size[0], img_size[1], bb)
            align_result = AlignResult(
                bounding_box=bb,
                landmarks=result['keypoints'])
            align_results.append(align_result)

        return align_results

    def close(self):
        self.mtcnn.__del__()        


