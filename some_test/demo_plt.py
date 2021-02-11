import matplotlib.pyplot as plt
import numpy as np

axis_x = np.array([-8, -7, -6, -5, -4, -3, -2, -1])
axis_y = np.array([0, 1, 2, 3, 4, 5, 6, 7])
fig1 = plt.figure(1)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.plot(axis_x, axis_y)
plt.draw()
plt.pause(4)    # 间隔的秒数： 4s
plt.close(fig1)

fig2 = plt.figure(2)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.plot(axis_y, axis_x)
plt.draw()
plt.pause(6)    # 间隔的秒数：6s
plt.close(fig1)

