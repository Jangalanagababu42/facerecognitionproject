import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


path = 'imageattendance'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodeList=[]
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode =face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList =f.readlines()
        nameList =[]
        for line in myDataList:
            entry=line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now =datetime.now()
            dtString=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
encodeListKnown = findEncodings(images)
print('encoding complete')

cap = cv2.VideoCapture(0)
while True:
    success,img =cap.read()
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)
    imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    facesCurFrame=face_recognition.face_locations(imgS)
    encodesCurFrame=face_recognition.face_encodings(imgS,facesCurFrame)

    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        facedis=face_recognition.face_distance(encodeListKnown,encodeFace)
        print(facedis)
        matchIndex = np.argmin(facedis)

        if matches[matchIndex]:
           name= classNames[matchIndex].upper()
           ##print(name)
           y1,x2,y2,x1 =faceLoc
           y1,x2,y2,x1 =y1*4,x2*4,y2*4,x1*4
           cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
           cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
           cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(225,225,225),2)
           markAttendance(name)

    cv2.imshow('Webcam',img)
    cv2.waitKey(1)




#aceLoc=face_recognition.face_locations(imgharish)[0]
#encodeharish=face_recognition.face_encodings(imgharish)[0]
#cv2.rectangle(imgharish,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)
#print(faceLoc)
#faceLoctest=face_recognition.face_locations(imgtest)[0]
#encodetest=face_recognition.face_encodings(imgtest)[0]
#cv2.rectangle(imgtest,(faceLoctest[3],faceLoctest[0]),(faceLoctest[1],faceLoctest[2]),(255,0,255),2)

#results=face_recognition.compare_faces([encodeharish],encodetest)
#facedis=face_recognition.face_distance([encodeharish],encodetest)
#print(results,facedis)

#imgharish = face_recognition.load_image_file('C:/Users/kushu/Desktop/propic/imagebasic/harish.jpg')
#imgharish = cv2.cvtColor(imgharish, cv2.COLOR_BGR2RGB)
#imgtest = face_recognition.load_image_file('C:/Users/kushu/Desktop/propic/imagebasic/testharish.jpg')
#imgtest = cv2.cvtColor(imgtest, cv2.COLOR_BGR2RGB)

