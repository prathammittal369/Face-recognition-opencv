import cv2 as cv
import numpy as np
import os

harcascade = cv.CascadeClassifier(r"path/to/ur/.xml_file")

people = os.listdir(r"directory/of/directories/of_photos_as_ur_data")

features = np.load(r"path/to/ur/features.npy/file", allow_pickle = True)
labels = np.load(r"path/to/ur/labels.npy/file")

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read(r"path/to/ur/face_reconize.yml/file")

#img = cv.imread(r"F:\Python\Vs code programming files\Imgs\Hawkeye\download.jpg")  was used as testing

video = cv.VideoCapture(r"F:\Python\Vs code programming files\4K HDR - Chitauri Invasion - The Avengers (2012).mp4") # was testing on this 😅
isTrue = True

while isTrue:
    isTrue,frame = video.read()
    grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    face_detect = harcascade.detectMultiScale(grey, scaleFactor = 1.1, minNeighbors = 20)


    for (x,y,w,h) in face_detect:
        face_crop = grey[y:y+h, x:x+w]

        label, confidence = face_recognizer.predict(face_crop)
        cv.rectangle(frame, (x,y), (x+w,y+h),(0,0,255), thickness = 1 )
        cv.putText(frame, people[label], (x,y-10), 1, 1.5, (0,0,255), 2)
    
    cv.imshow("Face", frame)

    if cv.waitKey(20) & 0XFF == ord('d'):
        break

video.release()
cv.destroyAllWindows()
