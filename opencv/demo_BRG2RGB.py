import cv2
import matplotlib.pyplot as plt


# opencv的接口使用BGR，而matplotlib.pyplot 则是RGB模式
def BGR2RGB(im):
    im=im.copy()
    temp=im[:,:,0].copy()
    im[:,:,0]=im[:,:,2].copy()
    im[:,:,2]=temp
    return im


im=cv2.imread("1.jpg")      # 读入彩色图片
plt.imshow(BGR2RGB(im))     # 显示彩色图片
plt.show()

