import cv2
import numpy as np
import face_recognition

imgharish = face_recognition.load_image_file('C:/Users/kushu/Desktop/propic/imagebasic/harish.jpg')
imgharish = cv2.cvtColor(imgharish,cv2.COLOR_BGR2RGB)
imgtest = face_recognition.load_image_file('C:/Users/kushu/Desktop/propic/imagebasic/testharish.jpg')
imgtest = cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)
faceLoc=face_recognition.face_locations(imgharish)[0]
encodeharish=face_recognition.face_encodings(imgharish)[0]
cv2.rectangle(imgharish,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)
#print(faceLoc)
faceLoctest=face_recognition.face_locations(imgtest)[0]
encodetest=face_recognition.face_encodings(imgtest)[0]
cv2.rectangle(imgtest,(faceLoctest[3],faceLoctest[0]),(faceLoctest[1],faceLoctest[2]),(255,0,255),2)

results=face_recognition.compare_faces([encodeharish],encodetest)
facedis=face_recognition.face_distance([encodeharish],encodetest)
print(results,facedis)
cv2.putText(imgtest,f'{results}{round(facedis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,225),2)

cv2.imshow('harish',imgharish)
cv2.imshow('harish test',imgtest)
cv2.waitKey(0)