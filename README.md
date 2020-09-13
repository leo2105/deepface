# deepface

![blackpink with deepface(vgg model)](./etc/example_blackpink.png)

Deep Learning Models for Face Detection/Recognition/Alignments, implemented in Tensorflow.


## Main Contributers

[@ildoonet](https://github.com/ildoonet) [@kimdwkimdw](https://github.com/kimdwkimdw) [@hahahahannie](https://github.com/hahahahannie) [@jaehobang](https://github.com/jaehobang)

## Models

### Baseline

A baseline model use dlib face detection module to crop rois. Then they will be forwarded at the vggface network to extract features.

- dlib face detection
- dlib face alignment
- VGGFace face recognition
- SSD Face Detection
  - Mobilenet V2
  - Training : https://github.com/ildoonet/deepface/tree/train/detector

## Experiments

### LFW Dataset

| Detector | Recognition Model | Description                    | Set        | 1-EER      | Accuracy |
|----------|-------------------|--------------------------------|------------|------------|----------|
|          | VGG               | Paper, No Embedding, Trained   | Test       | 0.9673     |          |
|          | VGG               | Paper, Embedding, Trained      | Test       | 0.9913     |          |
|          |                   |                                |            |            |          |
| dlib     | VGG               | no embedding, no training on lfw | Test       | 0.9400     | 0.936    |
| dlib     | VGG2-Resnet       | no training on lfw             | Test       | 0.9680     | 0.949    |
| SSD<br/>Mobilenet-v2 | VGG2-Resnet       | no training on lfw              | Test       | 0.990     | 0.973    |

## Install

### Requirements

```bash
$ pip install -r requirements.txt
```

### Install & Download Models

```bash
$ cd detectors/dlib
$ bash download.sh
$ cd ../ssd
$ bash download.sh
$ cd ../../recognizers/vggface/
$ bash download.sh
$ cd ../resnet/
$ bash download.sh
```

## Run

### Test on samples

```bash
$ python bin/face.py
```


## Reference

### Models

[1] VGG Face : http://www.robots.ox.ac.uk/~vgg/software/vgg_face/

[2] VGG Face in Tensorflow : https://github.com/ZZUTK/Tensorflow-VGG-face

[3] DLib : https://github.com/davisking/dlib

[4] Dlib Guide Blog : https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/

[5] VGG Face 2 Project : https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/

[6] Kera VGG Face2 : https://github.com/rcmalli/keras-vggface

### Datasets

[1] LFW : http://vis-www.cs.umass.edu/lfw/
