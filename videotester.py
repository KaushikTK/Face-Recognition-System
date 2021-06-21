# please place all the images that you want the recognizer to train on, inside a folder named as per the name variable in this code, inside the trainingImages folder

import cv2
import os
import numpy as np
import faceRecognition as fr

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("trainingData.yml") # loading the trained data

# name of the person
name = {
    0:"ABC", # name of the person inside folder 0
    1:"XYZ", # name of the person inside folder 1
    2:"PQR" # name of the person inside folder 2
    # you can keep adding as many folders as you want but update it here and train again incase you have added new images/folders
    }

cap = cv2.VideoCapture(0)

while True:
    ret,test_img = cap.read()
    faces_detected, gray_img = fr.faceDetection(test_img)
    
    for (x,y,w,h) in faces_detected:
        cv2.rectangle(test_img, (x,y), (x+w,y+h), (255,0,0), thickness = 5)
    
    resized_img = cv2.resize(test_img, (1000,1000))
    cv2.imshow("test picture", resized_img)
    cv2.waitKey(10)
    
    for face in faces_detected:
        (x,y,w,h) = face
        roi_gray = gray_img[y:y+h,x:x+h]
        
        label,confidence = face_recognizer.predict(roi_gray)
        print("confidence:", confidence)
        print("label:",label)
        fr.draw_rect(test_img, face)
        predicted_name = name[label]
        fr.put_text(test_img, predicted_name, x, y)
        
    resized_img = cv2.resize(test_img, (1000,1000))
    cv2.imshow("test picture", resized_img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows