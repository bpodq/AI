from PIL import Image       # 导入PIL库的Image类
import matplotlib.pyplot as plt


image = Image.open('1.jpg')   # 读取图像文件
image.show()                  # 显示图片
image.save('1_2.png')         # 保存这个图片文件对象


print(image.mode)        # RGB
print(image.size)
print(image.format)      # JPEG

