import cv2
import dlib
import mediapipe as mp
import landmarks_utils as lutils
import numpy as np
import imutils
import time
from landmarks_metrics import RegressionErrorCharacteristic 

# save the predicted values for the model 
y_pred=[]
landmark_points_68 = lutils.find_landmarks()
start = time.time()
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
frame = cv2.imread("./resources/indoor_005.png")
# save the actual values for the model (ground truth values)
y_true = lutils.read_landmarks("./resources/indoor_005.pts")

#detect the face
face_detector = dlib.get_frontal_face_detector()
frame = imutils.resize(frame,width=720)
gray =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
coordinates_bboxes= face_detector(gray,1)
for c in coordinates_bboxes:
        x_ini,y_ini,x_fin,y_fin = c.left(),c.top(),c.right(),c.bottom()
      
        print(x_ini)
        cv2.rectangle(frame,(x_ini,y_ini),(x_fin,y_fin),(0,0,255),4)
        cv2.waitKey(0)

print("Face detection successfully detected")

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
                     x = int(face_landmarks.landmark[index].x * width)
                     y = int(face_landmarks.landmark[index].y * height)
                     xvalue = face_landmarks.landmark[index].x * width
                     yvalue = face_landmarks.landmark[index].y * height
                     y_pred.append([xvalue, yvalue])
                    
                     cv2.circle(frame,( x, y),2,(255,255,0),2)
        cv2.imshow("Frame", frame)
print(len(landmark_points_68))
y_pred = np.array(y_pred, dtype=np.float32).reshape((-1, 2))

cv2.waitKey(0)
cv2.destroyAllWindows()


width_face= lutils.two_points_distance(x_ini,y_ini,x_fin,y_fin)
height_face= lutils.two_points_distance(x_ini,y_ini,x_fin,y_fin)

Df = lutils.calculate_DF(width_face,height_face)

# Calcute the value of NME
print("Normalized Mean Error")
print()
print(lutils.calculate_NME(y_true,y_pred,Df))
print()
# shOW CURVE CED Y AUC
myREC = RegressionErrorCharacteristic(y_true, y_pred)
myREC.plot_rec()
print("Failure Rate")
print()
print(lutils.calculate_fauilre_rate(y_true,y_pred,Df)/100)
end = time.time()
print(format(end-start))