import cv2
import os
import numpy as np
import faceRecognition as fr
#import matplotlib.pyplot as plt

test_img = cv2.imread('D:/Kaushik/Machine_Learning/ML Projects/Face_Recognition_System_using_LPDH/TestImages/1/IMG_20200501_122351.jpg')
faces_detected,gray_img = fr.faceDetection(test_img)
print("face_detected:",faces_detected)


'''
#run this when u r doing it for the first time.
#this saves the training in a file and hence we dont have to run it again. 

faces,faceID = fr.labels_for_training_data('D:/Kaushik/Machine_Learning/ML Projects/Face_Recognition_System_using_LPDH/trainingImages')
face_recognizer = fr.train_classifier(faces, faceID)
face_recognizer.save('D:/Kaushik/Machine_Learning/ML Projects/Face_Recognition_System_using_LPDH/trainingData.yml')

'''


#run this alone after training since we are saving all the information in a file after training.
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("trainingData.yml")




name = {0:"Roopkala", 1:"Kaushik", 2:"Kannan"}

for face in faces_detected:
    (x,y,w,h) = face
    roi_gray = gray_img[y:y+h,x:x+h]
    
    #confidence should be less(0-30) is good.
    label,confidence = face_recognizer.predict(roi_gray)
    print("confidence:", confidence)
    print("label:",label)
    fr.draw_rect(test_img, face)
    predicted_name = name[label]
    fr.put_text(test_img, predicted_name, x, y)


resized_img = cv2.resize(test_img, (1000,1000))
cv2.imshow("test picture", resized_img)
cv2.waitKey(0);
cv2.destroyAllWindows    
    
    

