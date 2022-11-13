# python drowniness_yawn.py --webcam 0 --alarm alarm.wav

from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
from statistics import mode
import tensorflow as tf
import numpy as np
import argparse
import imutils
import playsound
import time
import dlib
import cv2


def play_alarm_sound(path):
    playsound.playsound(path)


def launch_alarm():
    if args["alarm"] != "":
        t = Thread(target=play_alarm_sound,
                   args=(args["alarm"],))
        t.deamon = True
        t.start()


def eye_aspect_ratio(eye):

    A = dist.chebyshev(eye[1], eye[5])
    B = dist.chebyshev(eye[2], eye[4])
    C = dist.chebyshev(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)

    return ear


def eyes_infos(shape):
    """This method returns a tuple with three values as follow (ear,leftEye,rightEye)
    ear: Eye Aspect Ratio value
    leftEye, rightEye : left and right eyes 6 points cordinates"""

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)


def lip_distance(shape):
    """lip_distance method calculates the distance betwween the two lips"""
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance


def load_models(path):
    """loading predictive models for real-time predictions"""

    print("-> Laoding predective models .... ")
    model_vgg = tf.keras.models.load_model(path+"VGG_Model")
    model_vgg19 = tf.keras.models.load_model(path+"VGG19_Model")
    model_resnet = tf.keras.models.load_model(path+"Resnet")
    print("-> Models loaded sucessfully!")
    return [model_vgg, model_vgg19, model_resnet]


def most_common(List):
    """ mode() method is a statistics method that return the most frequent item or element in one set"""
    return(mode(List))


def judge_situation(models, frame):
    targets = ['Closed Eyes', 'Open Eyes', 'Normal', 'Yawning mouth']
    img_array = tf.keras.preprocessing.image.img_to_array(frame)
    img_batch = np.expand_dims(img_array, axis=0)
    predictions = []
    weights = [2, 3, 4]
    for model, weight in zip(models, weights):
        pred = model.predict(img_batch)
        pred = np.argmax(pred, axis=1)
        for index in range(weight):
            predictions.append(pred[0])
    # find out the the accurate prediction using 3 different models
    return targets[most_common(predictions)]


def save_image(target, img):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    cv2.imwrite("drowziness_snapshots/%s%s.jpg" %
                (target, timestr), img)


def screenshot(frame, middle_element):
    snapshot = frame[middle_element[1][1]-70:middle_element[1]
                     [1]+80, middle_element[1][0]-80:middle_element[1][0]+70]
    snapshot = cv2.resize(snapshot, (256, 256))
    return snapshot


ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
ap.add_argument("-a", "--alarm", type=str, default="",
                help="path alarm .WAV file")
args = vars(ap.parse_args())

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 20
YAWN_THRESH = 20
alarm_status = False
alarm_status2 = False
saying = False
COUNTER = 0
alarm_status = False
alarm_status2 = False
saying = False

print("-> Loading the predictor and detector...")
# detector = dlib.get_frontal_face_detector()
detector = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml")  # Faster but less accurate
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
predective_models = load_models('Drowsiness_models/')

print("-> Starting Video Stream")
vs = VideoStream(src=args["webcam"]).start()
# vs= VideoStream(usePiCamera=True).start()       //For Raspberry Pi
time.sleep(1.0)

while True:

    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector.detectMultiScale(gray, scaleFactor=1.1,
                                      minNeighbors=5, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)

    # for rect in rects:
    for (x, y, w, h) in rects:
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        eye = eyes_infos(shape)
        ear = eye[0]
        leftEye = eye[1]
        rightEye = eye[2]
        distance = lip_distance(shape)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        lip = shape[48:60]

        if ear < EYE_AR_THRESH:
            COUNTER += 1

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if alarm_status == False:
                    alarm_status = True
                    snapshot = screenshot(frame, leftEye)
                    save_image('snapshots', snapshot)
                    launch_alarm()
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    save_image('drowzy_frame', frame)
                    message = judge_situation(predective_models, snapshot)
                    print("the situation was : ", message)
                    time.sleep(1)

        else:
            COUNTER = 0
            alarm_status = False

        if (distance > YAWN_THRESH):
            cv2.putText(frame, "Yawn Alert", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if alarm_status2 == False and saying == False:
                alarm_status2 = True
                snapshot = screenshot(frame, lip)
                save_image('snapshots', snapshot)
                launch_alarm()
                save_image('yawning_frame', frame)
                message = judge_situation(predective_models, snapshot)
                print("the situation was : ", message)
                time.sleep(1)
        else:
            alarm_status2 = False

        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
