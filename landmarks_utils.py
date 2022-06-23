import math
from re import X
from tkinter import Y
import cv2
import dlib
import imutils
import numpy as np
import mediapipe as mp


def find_landmarks():
    landmark_points_68 = [162, 234, 93, 58, 172, 136, 149, 148, 152, 377, 378, 365, 397, 288, 323, 454, 389, 71, 63, 105, 66, 107, 336,
                          296, 334, 293, 301, 168, 197, 5, 4, 75, 97, 2, 326, 305, 33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373,
                          380, 61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181, 78, 82, 13, 312, 308, 317, 14, 87]
    return landmark_points_68


def read_landmarks(pts_file_path):
    points = []
    rows = open(pts_file_path).read().strip().split("\n")
    rows = rows[3:-1]  # take only the 68-landmarks
    for row in rows:
        # break the row into the filename and bounding box coordinates
        row = row.strip()  # remove blanks at the beginning and at the end
        row = row.split(" ")  # one space
        row = np.array(row, dtype="float32")  # convert list into float32
        (startX, startY) = row
        points.append([startX, startY])
        # points.extend(row)
    # convert a List into array of float32
    points = np.array(points, dtype=np.float32).reshape((-1, 2))  # (68, 2)
    return points


def two_points_distance(x_ini, y_ini, x_fin, y_fin):
    first_term = x_ini - x_fin
    second_term = y_ini - y_fin
    return math.sqrt(math.pow(first_term, 2) + math.pow(second_term, 2))


def calculate_DF(width_face, height_face):
    return math.sqrt(width_face*height_face)


def calculate_fauilre_rate(y_true, y_pred, df):
    NME = 0
    error = 0
    for i in range(0, 68):
        first_term = y_true[i, 0]-y_pred[i, 0]
        snd_term = y_true[i, 1]-y_pred[i, 1]
        NME = ((math.pow((first_term-snd_term), 2))/df)
        if NME >= 0.08:
            error = error+1

    return error


def calculate_NME(y_true, y_pred, df):
    NME = 0
    for i in range(0, 68):
        first_term = y_true[i, 0]-y_pred[i, 0]
        snd_term = y_true[i, 1]-y_pred[i, 1]
        NME = NME + ((math.pow((first_term-snd_term), 2))/df)

    return 1/68*NME


def Dlib_Model(imagen):
    cap = cv2.imread(imagen)
    # save the predicted values for the model
    y_pred = []
    face_detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    frame = imutils.resize(cap, width=1000)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    coordinates_bboxes = face_detector(gray, 1)
    for c in coordinates_bboxes:
        x_ini, y_ini, x_fin, y_fin = c.left(), c.top(), c.right(), c.bottom()
        shape = predictor(gray, c)
    for i in range(0, 68):
        x, y = shape.part(i).x, shape.part(i).y
        y_pred.append([x, y])
        
    width_face = two_points_distance(x_ini, y_ini, x_fin, y_fin)
    height_face = two_points_distance(x_ini, y_ini, x_fin, y_fin)
    Df = calculate_DF(width_face, height_face)
    y_pred = np.array(y_pred, dtype=np.float32).reshape((-1, 2))
    return y_pred , Df


def Mediapipe_Model(imagen):
# save the predicted values for the model 
  y_pred=[]
  landmark_points_68 = find_landmarks()
  mp_face_mesh = mp.solutions.face_mesh
  mp_drawing = mp.solutions.drawing_utils
  frame = cv2.imread(imagen)
  #detect the face
  face_detector = dlib.get_frontal_face_detector()
  frame = imutils.resize(frame,width=720)
  gray =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
  coordinates_bboxes= face_detector(gray,1)
  for c in coordinates_bboxes:
        x_ini,y_ini,x_fin,y_fin = c.left(),c.top(),c.right(),c.bottom()  
  
  with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5) as face_mesh:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height ,width,_ =frame_rgb.shape
        results = face_mesh.process(frame_rgb)
        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
                for index in landmark_points_68:
                     xvalue = face_landmarks.landmark[index].x * width
                     yvalue = face_landmarks.landmark[index].y * height
                     y_pred.append([xvalue, yvalue])

        print(len(landmark_points_68))
        y_pred = np.array(y_pred, dtype=np.float32).reshape((-1, 2))
        width_face= two_points_distance(x_ini,y_ini,x_fin,y_fin)
        height_face= two_points_distance(x_ini,y_ini,x_fin,y_fin)
        Df = calculate_DF(width_face,height_face)
        
        return y_pred, Df
