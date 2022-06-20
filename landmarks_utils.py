import math
from re import X
from tkinter import Y
import numpy as np
from sklearn.metrics import mean_squared_error


def find_landmarks():
    landmark_points_68 = [162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389,71,63,105,66,107,336,
                  296,334,293,301,168,197,5,4,75,97,2,326,305,33,160,158,133,153,144,362,385,387,263,373,
                  380,61,39,37,0,267,269,291,405,314,17,84,181,78,82,13,312,308,317,14,87]
    return landmark_points_68


def read_landmarks(pts_file_path):
    points = []
    rows = open(pts_file_path).read().strip().split("\n")
    rows = rows[3:-1] # take only the 68-landmarks
    for row in rows:
        # break the row into the filename and bounding box coordinates
        row = row.strip() # remove blanks at the beginning and at the end
        row = row.split(" ") # one space
        row = np.array(row, dtype="float32") # convert list into float32
        (startX, startY) = row
        points.append([startX, startY])
        # points.extend(row)
    # convert a List into array of float32
    points = np.array(points, dtype=np.float32).reshape((-1, 2)) # (68, 2)
    return points

def two_points_distance( x_ini,y_ini,x_fin,y_fin):
    first_term= x_ini - x_fin
    second_term= y_ini - y_fin
    return math.sqrt(math.pow(first_term,2) + math.pow(second_term,2))

def calculate_DF(width_face,height_face):
    return math.sqrt(width_face*height_face)

def calculate_fauilre_rate(y_true , y_pred , df):
    NME = 0
    error = 0
    for i in range(0,68):
         first_term = y_true[i,0]-y_pred[i,0]
         snd_term = y_true[i,1]-y_pred[i,1]
         NME = ((math.pow((first_term-snd_term),2))/df)
         if NME >=0.08:
            error = error+1


    return error

def calculate_NME(y_true , y_pred , df):
    NME = 0
    for i in range(0,68):
              first_term = y_true[i,0]-y_pred[i,0]
              snd_term = y_true[i,1]-y_pred[i,1]
              NME = NME + ((math.pow((first_term-snd_term),2))/df)
            
    return 1/68*NME
         
         

          