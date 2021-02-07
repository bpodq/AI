import matplotlib.pyplot as plt
import numpy as np
import cv2


image = cv2.imread('../image/panda.jpg', 0)
image2 = np.fliplr(image)
# plt.figure(figsize=(12, 8))
plt.imshow(image2)
plt.show()

