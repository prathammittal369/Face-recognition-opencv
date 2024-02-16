import cv2 as cv
import numpy as np
import os

harcascade = cv.CascadeClassifier(r"F:\Python\Vs code programming files\Opencv\Harcascade_face.xml")

people = os.listdir(r"F:\Python\Vs code programming files\Avengers train")

features = np.load(r"F:\Python\Vs code programming files\Features of avengers.npy", allow_pickle = True)
labels = np.load(r"F:\Python\Vs code programming files\Labels of avengers.npy")

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read(r"F:\Python\Vs code programming files\Face recognize5.yml")

#img = cv.imread(r"F:\Python\Vs code programming files\Imgs\Hawkeye\download.jpg")

video = cv.VideoCapture(r"F:\Python\Vs code programming files\4K HDR - Chitauri Invasion - The Avengers (2012).mp4")
isTrue = True

while isTrue:
    isTrue,frame = video.read()
    grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    face_detect = harcascade.detectMultiScale(grey, scaleFactor = 1.1, minNeighbors = 20)


    for (x,y,w,h) in face_detect:
        face_crop = grey[y:y+h, x:x+w]

        label, confidence = face_recognizer.predict(face_crop) # yaha label wahi label hai jo ki face_train file me labels me hai matalb ek naam ka ek index 0,1,2....
        cv.rectangle(frame, (x,y), (x+w,y+h),(0,0,255), thickness = 1 )
        cv.putText(frame, people[label], (x,y-10), 1, 1.5, (0,0,255), 2)
    
    cv.imshow("Face", frame)

    if cv.waitKey(20) & 0XFF == ord('d'):
        break

video.release()
cv.destroyAllWindows()
