from __future__ import absolute_import

import logging
import os
import time

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

import pickle
import sys
from glob import glob

import cv2
import numpy as np

import fire
from sklearn.metrics import roc_curve
from tqdm import tqdm
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
from deepface.confs.conf import DeepFaceConfs
from deepface.detectors.detector_dlib import FaceDetectorDlib
from deepface.detectors.detector_ssd import FaceDetectorSSDMobilenetV2, FaceDetectorSSDInceptionV2
from deepface.recognizers.recognizer_vgg import FaceRecognizerVGG
from deepface.recognizers.recognizer_resnet import FaceRecognizerResnet
from deepface.utils.common import get_roi, feat_distance_l2, feat_distance_cosine
from deepface.utils.visualization import draw_bboxs


class DeepFace:
    def __init__(self):
        self.detector = None
        self.recognizer = None

    def set_detector(self, detector):
        if self.detector is not None and self.detector.name() == detector:
            return
        if detector == FaceDetectorDlib.NAME:
            self.detector = FaceDetectorDlib()
        elif detector == 'detector_ssd_mobilenet_v2':
            self.detector = FaceDetectorSSDMobilenetV2()

    def set_recognizer(self, recognizer):
        if self.recognizer is not None and self.recognizer.name() == recognizer:
            return
        if recognizer == FaceRecognizerVGG.NAME:
            self.recognizer = FaceRecognizerVGG()
        elif recognizer == FaceRecognizerResnet.NAME:
            self.recognizer = FaceRecognizerResnet('redhot.pkl')

    def blackpink(self, visualize=True):
        imgs = ['./samples/blackpink/blackpink%d.jpg' % (i + 1) for i in range(7)]
        for img in imgs:
            self.run(image=img, visualize=visualize)

    def recognizer_test_run(self, detector=FaceDetectorDlib.NAME, recognizer=FaceRecognizerVGG.NAME, image='./samples/ajb.jpg', visualize=False):
        self.set_detector(detector)
        self.set_recognizer(recognizer)

        if isinstance(image, str):
            npimg = cv2.imread(image, cv2.IMREAD_COLOR)
        elif isinstance(image, np.ndarray):
            npimg = image
        else:
            sys.exit(-1)

        if npimg is None:
            sys.exit(-1)

        if recognizer:
            result = self.recognizer.detect([npimg[...,::-1]])
        return

    def run_recognizer(self, npimg, faces, recognizer=FaceRecognizerResnet.NAME):
        self.set_recognizer(recognizer)
        rois = []
        for face in faces:
            # roi = npimg[face.y:face.y+face.h, face.x:face.x+face.w, :]
            roi = get_roi(npimg, face, roi_mode=recognizer)
            if int(os.environ.get('DEBUG_SHOW', 0)) == 1:
                cv2.imshow('roi', roi)
                cv2.waitKey(0)
            rois.append(roi)
            face.face_roi = roi

        if len(rois) > 0:
            result = self.recognizer.detect(rois=rois, faces=faces)
            for face_idx, face in enumerate(faces):
                face.face_feature = result['feature'][face_idx]
                if result['name'][face_idx]:
                    name, score = result['name'][face_idx][0]
                    # if score < self.recognizer.get_threshold():
                    #     continue
                    face.face_name = name
                    face.face_score = score
        return faces

    # detector_ssd_mobilenet_v2
    def run(self, detector='detector_ssd_mobilenet_v2', recognizer=FaceRecognizerResnet.NAME, image='./samples/redhot/redhot_1.jpg',
            visualize=True):

        self.set_detector(detector)                                         # Inicializar modelo de deteccion
        self.set_recognizer(recognizer)                                     # Inicializar modelo de reconocimiento

        npimg = cv2.imread(image, cv2.IMREAD_COLOR)                         # Leer imagen

        if npimg is None:                                                   # Verificar lectura de imagen
            sys.exit(-1)

        t = time.time()
        faces = self.detector.detect(npimg)                                 # Correr detector de faces
        #print("Detecta {} faces en {} segundos".format(len(faces), round(time.time()-t,2)))

        if recognizer:
            t = time.time()
            faces = self.run_recognizer(npimg, faces, recognizer)           # Correr Reconocimiento de rostros
        #    print("Reconoce {} faces en {} segundos".format(len(faces), round(time.time()-t, 2)))

        img = draw_bboxs(np.copy(npimg), faces)                             # Dibujar los bounding boxes
        if visualize:
            f = 2
            cv2.imshow('DeepFace', cv2.resize(img, (img.shape[1]*f, img.shape[0]*f)))                                         # Mostrar imagen
            cv2.waitKey(0)

        return faces


    def save_features_path(self, path="./samples/blackpink/faces/"):
        """
        :param path: folder contain images("./samples/faces/")
        :return:
        """
        name_paths = [(os.path.basename(img_path).split('_')[0], img_path)
                      for img_path in glob(os.path.join(path, "*.*"))]

        features = {}
        for name, path in tqdm(name_paths):
            faces = self.run(image=path, visualize=False)
            features[name] = faces[0].face_feature

        import pickle
        with open('redhot.pkl', 'wb') as f:
            pickle.dump(features, f, protocol=2)


if __name__ == '__main__':
    faceRecognitionModel = DeepFace()
    if False:
        path = './samples/redhot/faces/'
        faceRecognitionModel.save_features_path(path)
    faces = faceRecognitionModel.run(image='./samples/redhot/redhot_1.jpg')
    [print(x) for x in faces]