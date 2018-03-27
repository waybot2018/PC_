import numpy as np
import cv2
#import serial

#ser = serial.Serial('COM4',9600, timeout=2)
face_cascade = cv2.CascadeClassifier('/home/pc/Загрузки/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('/home/pc/Загрузки/opencv-master/data/haarcascades/haarcascade_eye.xml')
#eye_cascade = cv2.CascadeClassifier('D:\\opencv\\build\\etc\\haarcascades\\haarcascade_eye.xml')



cap = cv2.VideoCapture(0)
while(True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
	cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
	x1=x+int(0.5*w)
	y1=y+int(0.5*h)
	print("x= ",x1,"y= ",y1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        kx=str(x)
        ky = str(y)
        cv2.putText(img,'x=' + kx, (0, 130), font, 1, (255,0,0), 2, cv2.LINE_AA)
        cv2.putText(img, 'y=' + ky, (0, 160), font, 1, (255,0,0), 2, cv2.LINE_AA)
        dx = bytes(kx, "ascii")
   #     ser.write(dx)
        #print(dx)

        roi_gray=gray[y:y+h,x:x+w]
        roi_color = img[y:y+h,x:x+w]

        smile=smile_cascade.detectMultiScale(roi_gray)
        #for(sx,sy,sw,sh)in smile:
            #cv2.putText(img,'SMILE', (0, 300), font, 3, (100,0,250), 5, cv2.LINE_AA)
           # cv2.rectangle(roi_color, (sx,sy), (sx+sw, sy+sh),(100,0,250),2)


      #  eyes = eye_cascade.detectMultiScale(roi_gray)
       # for(ex,ey,ew,eh)in eyes:
       #     cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh),(0,255,0),2)


    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
