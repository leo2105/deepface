import os
import sys

import dlib
import tensorflow as tf
import numpy as np
import cv2

from .detector_base import FaceDetector
from deepface.confs.conf import DeepFaceConfs
from deepface.utils.bbox import BoundingBox


class FaceDetectorSSD(FaceDetector):
    NAME = 'detector_ssd'

    def __init__(self, specific_model):
        super(FaceDetectorSSD, self).__init__()
        self.specific_model = specific_model
        graph_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            DeepFaceConfs.get()['detector'][self.specific_model]['frozen_graph']            # Pesos congelados del modelo
        )
        self.detector = self._load_graph(graph_path)                                        # Cargar Tensorflow graph

        self.tensor_image = self.detector.get_tensor_by_name('prefix/image_tensor:0')       # Tensor referente a imagen
        self.tensor_boxes = self.detector.get_tensor_by_name('prefix/detection_boxes:0')    # Tensor de bounding boxes
        self.tensor_score = self.detector.get_tensor_by_name('prefix/detection_scores:0')   # Tensor de scores
        self.tensor_class = self.detector.get_tensor_by_name('prefix/detection_classes:0')  # Tensor de clases detectadas

        predictor_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            DeepFaceConfs.get()['detector']['dlib']['landmark_detector']
        )
        self.predictor = dlib.shape_predictor(predictor_path)                               # Inicializar predictor de landmarks

        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))               # Uso eficiente gpu
        self.session = tf.Session(graph=self.detector, config=config)

    def _load_graph(self, graph_path):
        # We load the protobuf file from the disk and parse it to retrieve the
        # unserialized graph_def
        with tf.gfile.GFile(graph_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # Then, we import the graph_def into a new Graph and return it
        with tf.Graph().as_default() as graph:
            # The name var will prefix every op/nodes in your graph
            # Since we load everything in a new graph, this is not needed
            tf.import_graph_def(graph_def, name="prefix")
        return graph

    def name(self):
        return 'detector_%s' % self.specific_model

    def detect(self, npimg, resize=(480, 640)):

        height, width = npimg.shape[:2]
        infer_img = cv2.resize(npimg, resize, cv2.INTER_AREA) 

        dets, scores, classes = self.session.run([self.tensor_boxes, self.tensor_score, self.tensor_class], feed_dict={
            self.tensor_image: [infer_img]
        })
        dets, scores = dets[0], scores[0]

        faces = []
        for det, score in zip(dets, scores):
            if score < DeepFaceConfs.get()['detector'][self.specific_model]['score_th']: # th = 0.7 (deepface/config/basic.yml) 
                continue

            # Coordenadas del rostro
            y = int(max(det[0], 0) * height)                                             
            x = int(max(det[1], 0) * width)
            h = int((det[2] - det[0]) * height)
            w = int((det[3] - det[1]) * width)

            if w <= 1 or h <= 1:
                continue

            bbox = BoundingBox(x, y, w, h, score)

            # Calcular landmarks
            rect = dlib.rectangle(left=x, top=y, right=x + w, bottom=y + h)
            shape = self.predictor(npimg, rect)
            coords = np.zeros((68, 2), dtype=np.int)

            # Iterar sobre los 68 landmarks del rostro y convertirlos en duplas (x, y)
            for i in range(0, 68):
                coords[i] = (shape.part(i).x, shape.part(i).y)
            bbox.face_landmark = coords

            faces.append(bbox)

        faces = sorted(faces, key=lambda x: x.score, reverse=True)                       # Orderar por score 

        return faces


class FaceDetectorSSDInceptionV2(FaceDetectorSSD):
    def __init__(self):
        super(FaceDetectorSSDInceptionV2, self).__init__('ssd_inception_v2')


class FaceDetectorSSDMobilenetV2(FaceDetectorSSD):
    def __init__(self):
        super(FaceDetectorSSDMobilenetV2, self).__init__('ssd_mobilenet_v2')
