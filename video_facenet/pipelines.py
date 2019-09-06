from typing import List, cast
from functools import partial

from .facenet_types import Face
from .video import VideoProcessor
from .files import (
    bounding_box_file_name,
    landmarks_file_name,
    embeddings_file_name    
)

def generate_embeddings(encoder, faces:List[Face], **kwargs):
    images = [face.image for face in faces]
    embeddings = encoder.generate_embeddings(images)
    for face, embedding in zip(faces, embeddings):
        face.embedding = embedding
        face.image = None


def find_faces(pos, detector, image, faces: List[Face], **kwargs):
    found = cast(List[Face], detector.find_faces(image, detect_multiple_faces=True))
    for i, face in enumerate(found):
        face.pos = pos
        face.id = i
        faces.append(face)
    
class CsvWriter(object):
    def __init__(self, file_name, append=False):
        mode = "a" if append else "w"
        self.f = open(file_name, mode)
        self.new_line = "\n"

    def writeArray(self, *arr):
        line = ",".join([str(x) for x in arr])
        self.f.write(line)
        self.f.write(self.new_line)
    
    def flush(self):
        self.f.flush()

    def close(self):
        self.f.close()

class Saver(object):
    def __init__(self, save, writers):
        self.__save = save
        self.__writers = writers

    def __call__(self, **kwargs):
        self.__save(**kwargs)

    def flush(self):
        for writer in self.__writers:
            writer.flush()

    def close(self):
        for writer in self.__writers:
            writer.close() 


def write_bounding_boxes(writer, pos, id, face):
    writer.writeArray(pos, id, *face.bounding_box)

def write_landmarks(writer, pos, id, landmarks):
    for landmark in landmarks.items():
        writer.writeArray(pos, id, landmark[0], landmark[1][0], landmark[1][1])

def write_embeddings(writer, pos, id, embedding):
    writer.writeArray(pos, id, *[writer.embedding_format % x for x in embedding])

def save_faces(bounding_box_writer, landmarks_writer, embeddings_writer, faces:List[Face], **kwargs):              
    for face in faces:
        pos = face.pos
        id = face.id
        write_bounding_boxes(writer=bounding_box_writer, pos=pos, id=id, face=face)
        write_landmarks(writer=landmarks_writer, pos=pos, id=id, landmarks=face.landmarks)
        write_embeddings(writer=embeddings_writer, pos=pos, id=id, embedding=face.embedding)

def create_faces_saver(suffix="", embedding_format="%.6f", append=False):
    bounding_box_writer = CsvWriter(file_name=bounding_box_file_name(suffix), append=append)
    landmarks_writer = CsvWriter(file_name=landmarks_file_name(suffix), append=append)
    embeddings_writer = CsvWriter(file_name=embeddings_file_name(suffix), append=append)

    writers = [bounding_box_writer, landmarks_writer, embeddings_writer]

    embeddings_writer.embedding_format = embedding_format
    save = partial(save_faces, bounding_box_writer=bounding_box_writer, landmarks_writer=landmarks_writer, embeddings_writer=embeddings_writer)

    return Saver(save=save, writers=writers)

def process_video(video_path, model_path, start=0, suffix="", batch_size=64, end=None, **kwargs):
    from video_facenet.facenet import Detector, Facenet

    detector = Detector()
    encoder = Facenet(
        model_path=model_path,
        batch_size=batch_size
    )
    faces = []

    save = create_faces_saver(suffix=suffix)

    def process(save, pos, faces, batch_size, last, **kwargs):
        find_faces(faces=faces, pos=pos, **kwargs)
        if len(faces) >= batch_size or pos==last:
            generate_embeddings(faces=faces, **kwargs)
            save(faces=faces, **kwargs)
            save.flush()
            faces.clear()
            print("frame #", pos)

    video = VideoProcessor(video_path=video_path, detector=detector, encoder=encoder, save=save, faces=faces, batch_size=batch_size)
    video.iterate(process, start=start, end=end)
    save.close()
    detector.close()
    encoder.close()