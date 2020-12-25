import cv2
import numpy as np
import face_recognition

imgharish = face_recognition.load_image_file('imagebasic/harish.jpeg')
imgharish = cv2.cvtColor(imgharish,cv2.COLOR_BGR2RGB)
imgtest = face_recognition.load_image_file('imagebasic/harish test.jpeg')
imgtest = cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)





cv2.imshow('harish',imgharish)
cv2.imshow('harish test',imgtest)
cv2.waitkey(0)