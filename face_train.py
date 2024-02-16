import cv2 as cv
import numpy as np
import os

ls = os.listdir(r"F:\Python\Vs code programming files\Imgs2")
Dir = r"F:\Python\Vs code programming files\Imgs2"

features = []
labels = []

harcascade = cv.CascadeClassifier(r'F:\Python\Vs code programming files\Opencv\Harcascade_face.xml')

def create_train():

    for avenger in ls:
        path = os.path.join(Dir,avenger)
        label = ls.index(avenger)

        for img in os.listdir(path):

            img_path = os.path.join(path,img)
            img_read = cv.imread(img_path)
            grey = cv.cvtColor(img_read, cv.COLOR_BGR2GRAY)

            face_detct = harcascade.detectMultiScale(grey, scaleFactor = 1.1, minNeighbors = 10)

            for (x,y,w,h) in face_detct:
                crop_face = grey[y:y+h, x:x+w]
                features.append(crop_face)
                labels.append(label)
                

create_train()

features = np.array(features, dtype = "object")
labels = np.array(labels)

""" print("Trained!!!!!!!!!!!!!! 👍👍👍👍👍👍") """

face_train = cv.face.LBPHFaceRecognizer_create()
face_train.train(features,labels)
face_train.save("Face Recognizer.yml")

np.save("Features.npy",features)
np.save("Labels.npy",labels)
print(labels)
