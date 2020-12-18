from argparse import ArgumentParser
import numpy as np
import cv2
import onnxruntime
import sys
from pathlib import Path
from collections import deque
import datetime
import pandas as pd
import random
from os import listdir
from os.path import isfile, join

#local imports
from face_detector import FaceDetector, ResizeWithAspectRatio
from utils import draw_axis
from NodeShakeMode import NodShakeMode, NodShakeHMM


root_path = str(Path(__file__).absolute().parent)

MAXLEN = 9 * 10  # max length of deque object used for tracking movement
# my mac currently has 9 frames per second

EULER_ANGLES = ["yaw", "pitch", "roll"]
HEAD_POSE = ["nod", "shake", "other"]

class HeadMovementTracker(object):
    def __init__(self,):
        self.hmm_model = NodShakeHMM(maxlen=7)

        self.face_d = FaceDetector()

        self.eye_model = f'{root_path}/pretrained/haarcascades/haarcascade_eye_tree_eyeglasses.xml'
        self.sess = onnxruntime.InferenceSession(f'{root_path}/pretrained/fsanet-1x1-iter-688590.onnx')
        self.sess2 = onnxruntime.InferenceSession(f'{root_path}/pretrained/fsanet-var-iter-688590.onnx')

        self.font = cv2.FONT_HERSHEY_SIMPLEX

        self.prev_head_pose_hmm = 'stationary'

    def track_head_movement(self, frame):
        #get face bounding boxes from frame
        face_bb, data = self.face_d.detect_face_and_eyes_enhanced(frame, cv2.CascadeClassifier(self.eye_model))

        for (x1, y1, x2, y2) in face_bb:
            if x1 <= 0 or y1 <= 0 or x2 <= 0 or y2 <= 0:
                continue
            face_roi = frame[y1:y2+1,x1:x2+1]

            #preprocess headpose model input
            try:
                face_roi = cv2.resize(face_roi,(64,64))
            except:
                print(x1, y1, x2, y2)
                continue
            face_roi = face_roi.transpose((2,0,1))
            face_roi = np.expand_dims(face_roi,axis=0)
            face_roi = (face_roi-127.5)/128
            face_roi = face_roi.astype(np.float32)  # -> (1, 3, 64, 64)

            #get headpose
            res1 = self.sess.run(["output"], {"input": face_roi})[0]
            res2 = self.sess2.run(["output"], {"input": face_roi})[0]

            yaw, pitch, roll = np.mean(np.vstack((res1, res2)), axis=0)

            data.add_euler_angles(yaw, pitch, roll)
            if data.x1:
                self.hmm_model.add_data(data)
            new_head_pose_hmm = self.hmm_model.determine_pose()
            if new_head_pose_hmm == 'stationary' or self.prev_head_pose_hmm == new_head_pose_hmm:
                head_pose_hmm = new_head_pose_hmm
            else:
                head_pose_hmm = self.prev_head_pose_hmm
            self.prev_head_pose_hmm = new_head_pose_hmm

            draw_axis(frame, yaw, pitch, roll, tdx=(x2-x1)//2+x1, tdy=(y2-y1)//2+y1, size=50)

            cv2.putText(frame, head_pose_hmm, (x1, y1), self.font, 2, (255, 255, 255), 3, cv2.LINE_AA)

            #draw face bb
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return frame    
