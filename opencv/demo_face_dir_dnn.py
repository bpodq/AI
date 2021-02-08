import cv2
import dlib
import os


# 谨慎，花费时间较长！

path = 'D:/Python37/Lib/site-packages/cv2/data/'

# 1.加载文件和图片 2.进行灰度处理 3.得到haar特征 4.检测人脸 5.标记
face_xml = cv2.CascadeClassifier(path + 'haarcascade_frontalface_default.xml')
eye_xml = cv2.CascadeClassifier(path + 'haarcascade_eye.xml')

dnnFaceDetector = dlib.cnn_face_detection_model_v1("../FaceDetection/mmod_human_face_detector.dat")

imgpath = '../image/'
for file in os.listdir(imgpath):
    file_path = os.path.join(imgpath, file)
    if not os.path.isdir(file_path):
        print(file_path)
        img = cv2.imread(imgpath + file)
        cv2.imshow(file, img)

        result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        rects = dnnFaceDetector(result, 1)

        for (i, rect) in enumerate(rects):
            x1 = rect.rect.left()
            y1 = rect.rect.top()
            x2 = rect.rect.right()
            y2 = rect.rect.bottom()

            # Rectangle around the face
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 3)

        cv2.imshow('result', img)
        cv2.imwrite(imgpath+'dnn_result/'+file, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

