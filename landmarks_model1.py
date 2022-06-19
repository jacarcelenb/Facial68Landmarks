
import cv2
import dlib
import imutils
import numpy as np
import landmarks_utils as lutils
import time
from landmarks_metrics import RegressionErrorCharacteristic 
start = time.time()
cap = cv2.imread("./resources/indoor_005.png")

# detecto facial
# save the predicted values for the model 
y_pred=[]
# save the actual values for the model (ground truth values)
y_true = lutils.read_landmarks("./resources/indoor_005.pts")

face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

frame = imutils.resize(cap,width=1000)
gray =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
coordinates_bboxes= face_detector(gray,1)
for c in coordinates_bboxes:
        x_ini,y_ini,x_fin,y_fin = c.left(),c.top(),c.right(),c.bottom()
      
        print(x_ini)
        cv2.rectangle(frame,(x_ini,y_ini),(x_fin,y_fin),(0,255,0),4)
        cv2.waitKey(0)
        shape = predictor(gray,c)
        for i in range(0,68):
            x ,y  =shape.part(i).x ,shape.part(i).y
            y_pred.append([x, y])
           
            cv2.circle(frame,(x,y),2,(255,0,0),-1)
            cv2.putText(frame,str(i+1),(x,y-5),1,0.8,(0,255,255),1)
cv2.imshow("Frame",frame)
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