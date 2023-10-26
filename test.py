import time

import ImageProcessing
import readData
import cv2
import numpy as np

time.sleep(2)
data=readData.gameData()
for i in range(30):
    img=data.grabScreen()
    img=ImageProcessing.ImageProcessing(img)
    cv2.imwrite('test'+str(i)+".jpg",img)
    time.sleep(0.2)