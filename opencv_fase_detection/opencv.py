import numpy as np
import cv2
import dlib
import imutils
import argparse
import time
from imutils import face_utils
from imutils.video import VideoStream
#import serial

#ser = serial.Serial('COM4',9600, timeout=2)

def rect_to_bb(rect):
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	return (x, y, w, h)

def index_max(array, n): 
	array_new=[]
	for i in range(len(array)):
		array_new.append(array[i][n])
		maximym = max(array_new)
		indexmax=array_new.index(maximym)
		return maximym, indexmax

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--picamera", type=int, default=-1,
	help="использование pi камеры")
args = vars(ap.parse_args())



detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


profileface = cv2.CascadeClassifier('haarcascade_profileface.xml')


vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)

while(True):

	frame = vs.read() 
	frame = imutils.resize(frame, width=600)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
	rects = detector(gray, 0)
	
	faces_profil = profileface.detectMultiScale(gray, 1.3, 5)


	if(len(rects) >= 0):
		#print(len(rects))
		all_face=[]
		#print(rects)
		for rect in rects:
			#shape_cam = predictor(gray, rect)
			#shape = face_utils.shape_to_np(shape_cam)
			(x, y, w, h) = face_utils.rect_to_bb(rect)
			face= [x, y, w, h]
			all_face.append(face)
			
			#if (x>0 and y>0 and x+h<600 and y+h<400):
				#cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
				#face_descriptor= facerec.compute_face_descriptor(frame, shape_cam)

			#for (x, y) in shape:
				#cv2.circle(frame, (x, y), 0, (0, 255, 255), -1)
				#cv2.imshow("Frame", frame)

		if(len(all_face)>0):
			maximym, indexmax = index_max(all_face, 2)
			x_index=all_face[indexmax][0]
			y_index=all_face[indexmax][1]
			w_index=all_face[indexmax][2]
			h_index=all_face[indexmax][3]
			cv2.rectangle(frame,(x_index,y_index),(x_index+w_index,y_index+h_index),(255,0,0),2)
			cv2.imshow("Frame", frame)

	


	if(len(rects)==0):
		all_face_profil=[]
		for (x, y, w, h) in faces_profil:
			face= [x, y, w, h]
			all_face_profil.append(face)


		if(len(all_face_profil)>0):
			maximym, indexmax = index_max(all_face_profil, 2)
			x_index=all_face_profil[indexmax][0]
			y_index=all_face_profil[indexmax][1]
			w_index=all_face_profil[indexmax][2]
			h_index=all_face_profil[indexmax][3]
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
			cv2.imshow("Frame", frame)



	x1=x_index+int(0.5*w)
	y1=y_index+int(0.5*h)
	print("x= ",x1,"y= ",y1)
	#font = cv2.FONT_HERSHEY_SIMPLEX

	cv2.imshow("Frame", frame)

	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
