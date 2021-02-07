import cv2
import matplotlib.pyplot as plt
import dlib
import imutils

img = cv2.imread("1.jpg")
plt.imshow(imutils.opencv2matplotlib(img))
plt.show()

