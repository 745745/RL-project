import os
import re
import cv2 # opencv library
import numpy as np
from os.path import isfile, join
import matplotlib.pyplot as plt


def ImageProcessing(img):
    img=np.where(np.logical_and(img>100,img<180), 0,img )
    return img