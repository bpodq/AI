import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


path = 'D:\\Python37\\Lib\\site-packages\\cv2\\data\\'

# 1.加载文件和图片 2.进行灰度处理 3.得到haar特征 4.检测人脸 5.标记
face_xml = cv2.CascadeClassifier(path + 'haarcascade_frontalface_default.xml')
eye_xml = cv2.CascadeClassifier(path + 'haarcascade_eye.xml')


imgpath = 'image'
for root, dirs, files in os.walk(imgpath):
    for file in files:
        img = cv2.imread(imgpath + '\\' + file)
        cv2.imshow('img', img)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1.灰色图像 2.缩放系数 3.目标大小
        faces = face_xml.detectMultiScale(gray, 1.3, 5)
        print('face = ', len(faces))
        print(faces)

        # 绘制人脸，为人脸画方框
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x + w, y + h), (255, 0, 0), 2)
            roi_face = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_xml.detectMultiScale(roi_face)
            print('eyes = ', len(eyes))
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color, (ex,ey),(ex + ew, ey + eh), (0,255,0), 2)

        cv2.imshow('dat', img)
        cv2.waitKey(0)

