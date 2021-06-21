# please place all the images that you want the recognizer to train on, inside a folder name as per the name variable in this code, inside the trainingImages folder

import cv2
import os
import numpy as np
import faceRecognition as fr

test_img = cv2.imread('path of your image file inside testImages folder')
faces_detected,gray_img = fr.faceDetection(test_img)
print("face_detected:",faces_detected)


#run this when u r doing it for the first time.
#this saves the training in a file and hence we dont have to run it again. 
faces,faceID = fr.labels_for_training_data('trainingImages')
face_recognizer = fr.train_classifier(faces, faceID)
face_recognizer.save('trainingData.yml')


#run this alone after training since we are saving all the information in a file after training.
#face_recognizer = cv2.face.LBPHFaceRecognizer_create()
#face_recognizer.read("trainingData.yml")


# name of the person
name = {
    0:"ABC", # name of the person inside folder 0
    1:"XYZ", # name of the person inside folder 1
    2:"PQR" # name of the person inside folder 2
    # you can keep adding as many folders as you want but update it here and train again incase you have added new images/folders
    }

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
cv2.waitKey(0)
cv2.destroyAllWindows