from tkinter import N, Y
from tkinter.tix import Y_REGION
import landmarks_utils as lutlis

# Model library Dlib
# ground truth
y_true =[]
# predicted values
y_pred =[]
# DF distance
Df = 0
error = 0
for i in range(44, 55):
    y_true = lutlis.read_landmarks("./dataset/y_true/indoor_0"+str(i)+".pts")
    y_pred , Df = lutlis.Mediapipe_Model("./dataset/y_pred/indoor_0"+str(i)+".png")
    print("NME de "+"indoor_0"+str(i)+".pts")
    NME = lutlis.calculate_NME(y_true,y_pred,Df)
    print(NME)

    if NME >= 0.08:
        error = error + 1

print("Tasa de fallos")
print((error)/100)

