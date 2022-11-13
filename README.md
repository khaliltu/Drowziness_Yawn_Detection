# Drowsiness and Yawn detection with voice alert using Dlib & Tensorflow

# Real time Detection - Deep Learning (CNN) - Tranfer Learning (VGG16/VGG19/Resnet50) - Ensemble Learning (Bagging)

Python Application to detect Drowsiness and Yawn and alert the user using Dlib and Tensorflow.

## Dependencies

1. Python 3
2. opencv
3. dlib
4. imutils
5. scipy
6. numpy
7. argparse
8. tensorflow

## Run

```
Python3 drowsiness_yawn.py -- webcam 0 --alarm alarm.wav	//For external webcam, use the webcam number accordingly
```

## Setups

Change the threshold values according to your need

```
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 20
YAWN_THRESH = 20	//change this according to the distance from the camera
```

## Authors

**Khalil Turki**

## This work is an amelioration and expansion of one older version established by:

**Arijit Das**

## Kaggle Dataset link for training models:

https://www.kaggle.com/datasets/serenaraju/yawn-eye-dataset-new
