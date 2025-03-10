"""
Main program to run the detection and TCP
"""
from facial_landmark import FaceMeshDetector
from argparse import ArgumentParser
import cv2
import mediapipe as mp
import numpy as np
import os
clear = lambda: os.system('cls')
# for TCP connection with unity
import socket
import pickle

# face detection and facial landmark


# pose estimation and stablization
from pose_estimator import PoseEstimator
from stabilizer import Stabilizer

# Miscellaneous detections (eyes/ mouth...)
from facial_features import FacialFeatures, Eyes

import sys

# global variable
port = 5066         # have to be same as unity

# init TCP connection with unity
# return the socket connected
def init_TCP():
    port = args.port

    # '127.0.0.1' = 'localhost' = your computer internal data transmission IP
    address = ('127.0.0.1', port)
    # address = ('192.168.0.107', port)

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(address)
        # print(socket.gethostbyname(socket.gethostname()) + "::" + str(port))
        print("Connected to address:", socket.gethostbyname(socket.gethostname()) + ":" + str(port))
        return s
    except OSError as e:
        print("Error while connecting :: %s" % e)
        
        # quit the script if connection fails (e.g. Unity server side quits suddenly)
        sys.exit()

    # s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # # print(socket.gethostbyname(socket.gethostname()))
    # s.connect(address)
    # return s

def send_info_to_unity(s, args):
    msg = '%.4f ' * len(args) % args

    try:
        s.send(bytes(msg, "utf-8"))
    except socket.error as e:
        print("error while sending :: " + str(e))

        # quit the script if connection fails (e.g. Unity server side quits suddenly)
        sys.exit()

def print_debug_msg(args):
    msg = '%.4f ' * len(args) % args
    print(msg)

def main():
    cap = cv2.VideoCapture(args.cam)
    detector = FaceMeshDetector()
    success, img = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        return
    pose_estimator = PoseEstimator((img.shape[0], img.shape[1]))
    image_points = np.zeros((pose_estimator.model_points_full.shape[0], 2))
    iris_image_points = np.zeros((10, 2))
    pose_stabilizers = [Stabilizer(state_num=2, measure_num=1, cov_process=0.1, cov_measure=0.1) for _ in range(6)]
    eyes_stabilizers = [Stabilizer(state_num=2, measure_num=1, cov_process=0.1, cov_measure=0.1) for _ in range(6)]
    mouth_dist_stabilizer = Stabilizer(state_num=2, measure_num=1, cov_process=0.1, cov_measure=0.1)

    if args.connect:
        socket = init_TCP()
        with open('params/fitting_params_male.pkl', 'rb') as f:
            data = pickle.load(f)
        send_info_to_unity(socket, (1.1111, *data["betas"][:300]))

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        img_facemesh, faces, recog = detector.findFaceMesh(img)
        
        print("faces:", faces)  # Debugging line to check structure

        img = cv2.flip(img, 1)

        if isinstance(faces, list) and len(faces) > 0:
            if len(faces) >= 468:  # Ensure there are enough landmarks
                for i in range(len(image_points)):
                    image_points[i, 0] = faces[i][0]
                    image_points[i, 1] = faces[i][1]

                for j in range(len(iris_image_points)):
                    iris_image_points[j, 0] = faces[j + 468][0]
                    iris_image_points[j, 1] = faces[j + 468][1]

                pose = pose_estimator.solve_pose_by_all_points(image_points)

                x_ratio_left, y_ratio_left = FacialFeatures.detect_iris(image_points, iris_image_points, Eyes.LEFT)
                x_ratio_right, y_ratio_right = FacialFeatures.detect_iris(image_points, iris_image_points, Eyes.RIGHT)

                ear_left = FacialFeatures.eye_aspect_ratio(image_points, Eyes.LEFT)
                ear_right = FacialFeatures.eye_aspect_ratio(image_points, Eyes.RIGHT)

                pose_eye = [ear_left, ear_right, x_ratio_left, y_ratio_left, x_ratio_right, y_ratio_right]

                mar = FacialFeatures.mouth_aspect_ratio(image_points)
                mouth_distance = FacialFeatures.mouth_distance(image_points)

                steady_pose = []
                pose_np = np.array(pose).flatten()
                for value, ps_stb in zip(pose_np, pose_stabilizers):
                    ps_stb.update([value])
                    steady_pose.append(ps_stb.state[0])

                steady_pose = np.reshape(steady_pose, (-1, 3))

                steady_pose_eye = []
                for value, ps_stb in zip(pose_eye, eyes_stabilizers):
                    ps_stb.update([value])
                    steady_pose_eye.append(ps_stb.state[0])

                mouth_dist_stabilizer.update([mouth_distance])
                steady_mouth_dist = mouth_dist_stabilizer.state[0]

                roll = np.clip(np.degrees(steady_pose[0][1]), -90, 90)
                pitch = np.clip(-(180 + np.degrees(steady_pose[0][0])), -90, 90)
                yaw = np.clip(np.degrees(steady_pose[0][2]), -90, 90)

                if args.connect:
                    send_info_to_unity(socket, (9.9999, roll, pitch, yaw, mar))

                if args.debug:
                    log_params(roll, pitch, yaw, ear_left, ear_right, x_ratio_left, y_ratio_left, x_ratio_right, y_ratio_right, mar, mouth_distance)

                pose_estimator.draw_axes(img_facemesh, steady_pose[0], steady_pose[1])

        else:
            pose_estimator = PoseEstimator((img_facemesh.shape[0], img_facemesh.shape[1]))

        try:
            if args.connect:
                if len(recog) > 0:
                    send_info_to_unity(socket, (8.8888, float(recognition2int(recog[0][0].category_name))))
                else:
                    send_info_to_unity(socket, (8.8888, -1.0))
        except Exception as e:
            print(e)

        cv2.imshow('Facial landmark', img_facemesh)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()


def recognition2int(recog_text:str):
    recog_list = ["Pointing_Up", "Open_Palm", "Closed_Fist", "Victory", "ILoveYou","Thumb_Down", "Thumb_Up"]
    if recog_text in recog_list:
        return recog_list.index(recog_text)
    else:
        return -1

def log_params(roll, pitch, yaw,
                    ear_left, ear_right, x_ratio_left, y_ratio_left, x_ratio_right, y_ratio_right,
                    mar, mouth_distance):
    print("roll          : " + "="*(int(roll)+45))
    print("pitch         : " + "="*(int(pitch)+45))
    print("yaw           : " + "="*(int(yaw)+45))
    print("ear_left      : " + "="*(int(ear_left*100)+45))
    print("ear_right     : " + "="*(int(ear_right*100)+45))
    print("x_ratio_left  : " + "="*(int(x_ratio_left*100)+45))
    print("y_ratio_left  : " + "="*(int(y_ratio_left*100)+45))
    print("x_ratio_right : " + "="*(int(x_ratio_right*100)+45))
    print("y_ratio_right : " + "="*(int(y_ratio_right*100)+45))
    print("mar           : " + "="*(int(mar*45)))
    print("mouth_distance: " + "="*(int(mouth_distance*10)+45))
    
if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--connect", action="store_true",
                        help="connect to unity character",
                        default=False)

    parser.add_argument("--port", type=int, 
                        help="specify the port of the connection to unity. Have to be the same as in Unity", 
                        default=5066)

    parser.add_argument("--cam", type=int,
                        help="specify the camera number if you have multiple cameras",
                        default=0)

    parser.add_argument("--debug", action="store_true",
                        help="showing raw values of detection in the terminal",
                        default=False)

    args = parser.parse_args()

    # demo code
    main()
