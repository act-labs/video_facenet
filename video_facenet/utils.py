from typing import List
import numpy as np
from .facenet_types import Image, Landmarks, AlignResult

def fixed_standardize(image: Image) -> Image:
    image = image - 127.5
    image = image / 128.0
    return image

def crop(image: Image, bounding_box: List[int], margin: float) -> Image:
    """
    img = image from misc.imread, which should be in (H, W, C) format
    bounding_box = pixel coordinates of bounding box: (x0, y0, x1, y1)
    margin = float from 0 to 1 for the amount of margin to add, relative to the
        bounding box dimensions (half margin added to each side)
    """

    if margin < 0:
        raise ValueError("the margin must be a value between 0 and 1")
    if margin > 1:
        raise ValueError(
            "the margin must be a value between 0 and 1 - this is a change from the existing API")

    img_height = image.shape[0]
    img_width = image.shape[1]
    x_0, y_0, x_1, y_1 = bounding_box[:4]
    margin_height = (y_1 - y_0) * margin / 2
    margin_width = (x_1 - x_0) * margin / 2
    x_0 = int(np.maximum(x_0 - margin_width, 0))
    y_0 = int(np.maximum(y_0 - margin_height, 0))
    x_1 = int(np.minimum(x_1 + margin_width, img_width))
    y_1 = int(np.minimum(y_1 + margin_height, img_height))
    return image[y_0:y_1, x_0:x_1, :], (x_0, y_0, x_1, y_1)        

def fix_mtcnn_bb(max_y: int, max_x: int, bounding_box: List[int]) -> List[int]:
    """ mtcnn results can be out of image so fix results
    """
    x1, y1, dx, dy = bounding_box[:4]
    x2 = x1 + dx
    y2 = y1 + dy
    x1 = max(min(x1, max_x), 0)
    x2 = max(min(x2, max_x), 0)
    y1 = max(min(y1, max_y), 0)
    y2 = max(min(y2, max_y), 0)
    return [x1, y1, x2, y2]        

def preprocess(
        image: Image,
        desired_height: int,
        desired_width: int,
        margin: float,
        bbox: List[int]=None,
        landmark: Landmarks=None,
        use_affine: bool=False):
    image_height, image_width = image.shape[:2]
    margin_height = int(desired_height + desired_height * margin)
    margin_width = int(desired_width + desired_width * margin)
    M = None
    if landmark is not None and use_affine:
        M = get_transform_matrix(landmark['left_eye'],
                                 landmark['right_eye'],
                                 (0.35, 0.35),
                                 desired_height,
                                 desired_width,
                                 margin)

    if bbox is None:
        # use center crop
        bbox = [0, 0, 0, 0]
        bbox[0] = int(image_height * 0.0625)
        bbox[1] = int(image_width * 0.0625)
        bbox[2] = image.shape[1] - bbox[0]
        bbox[3] = image.shape[0] - bbox[1]
    if M is None:
        cropped = crop(image, bbox, margin)[0]
        return cropped
    else:
        # do align using landmark
        warped = cv2.warpAffine(
            image, M, (margin_height, margin_width), flags=cv2.INTER_CUBIC)
        return warped        

def get_center_box(img_size: np.ndarray, results: List[AlignResult]):
    # x1, y1, x2, y2
    all_bbs = np.asarray([result.bounding_box for result in results])
    all_landmarks = [result.landmarks for result in results]
    bounding_box_size = (all_bbs[:, 2] - all_bbs[:, 0]) * \
        (all_bbs[:, 3] - all_bbs[:, 1])
    img_center = img_size / 2
    offsets = np.vstack([(all_bbs[:, 0] + all_bbs[:, 2]) / 2 - img_center[1],
                         (all_bbs[:, 1] + all_bbs[:, 3]) / 2 - img_center[0]])
    offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
    index = np.argmax(
        bounding_box_size -
        offset_dist_squared *
        2.0)  # some extra weight on the centering
    out_bb = all_bbs[index, :]
    out_landmark = all_landmarks[index] if index < len(all_landmarks) else None
    align_result = AlignResult(bounding_box=out_bb, landmarks=out_landmark)
    return [align_result]        
