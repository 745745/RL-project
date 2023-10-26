import ctypes

import win32api
import win32gui
import win32ui
import win32process
from scipy.ndimage import zoom
import cv2
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import *
import sys
import time
import qimage2ndarray
import numpy as np
from PIL import Image


class gameData:
    def __init__(self):
        self.time_offset = [0x14, 0x60, 0x10, 0x58]
        self.health_offset = [0x10, 0x60, 0x10, 0x7C]
        self.flash_offset = [0xC, 0x60, 0x10, 0xAC]
        handle = win32gui.FindWindow(None, "DodgeShow")

        process_id = win32process.GetWindowThreadProcessId(handle)[1]

        self.process_handle = win32api.OpenProcess(0x1F0FFF, False, process_id)

        self.kernel = ctypes.windll.LoadLibrary(r"C:\Windows\System32\kernel32.dll")
        self.lastData = [0.0, 1500.0, 0.0]

    def grabScreen(self, mode="normal"):
        hwnd = win32gui.FindWindow(None, "DodgeShow")
        app = QApplication(sys.argv)
        screen = QApplication.primaryScreen()
        point = (self.readPosition()[0], self.readPosition()[1])
        img = screen.grabWindow(hwnd).toImage()
        img.convertTo(QImage.Format_Grayscale8)
        img = qimage2ndarray.byte_view(img)[:, :, 0]
        x_min=point[0]-50 if point[0]-50>0 else 0
        y_min=point[1]-85 if point[1]-85>0 else 0
        img[y_min:point[1]+10,x_min:point[0]]=255
        img = cv2.resize(img, (224, 224))
        if mode=="VGG":
            img = cv2.merge((img, img, img))
            img = img[:, :,:, np.newaxis]
        else:
            img=img[:, :, np.newaxis]
        return img
        # test
        '''''
        image = Image.fromarray(img, 'L')
        image.save("test.jpg")
        image.show()
        '''''

    def grabScreen_test(self):
        hwnd = win32gui.FindWindow(None, "DodgeShow")
        app = QApplication(sys.argv)
        screen = QApplication.primaryScreen()
        point = (self.readPosition()[0], self.readPosition()[1])
        img = screen.grabWindow(hwnd).toImage()
        img.convertTo(QImage.Format_Grayscale8)
        img = qimage2ndarray.byte_view(img)[:, :, 0]
        x_min=point[0]-50 if point[0]-50>0 else 0
        y_min=point[1]-85 if point[1]-85>0 else 0
        img[y_min:point[1],x_min:point[0]]=255
        img = cv2.resize(img, (224, 224))
        cv2.imwrite('test1.jpg',img)
        return img


    def readTime(self):
        base_address = ctypes.c_long(0x00883DB4)
        self.kernel.ReadProcessMemory(int(self.process_handle), base_address.value, ctypes.byref(base_address), 4, None)
        for i in self.time_offset:
            self.kernel.ReadProcessMemory(int(self.process_handle), base_address.value + i, ctypes.byref(base_address),
                                          4, None)

        time = ctypes.c_double()
        self.kernel.ReadProcessMemory(int(self.process_handle), base_address, ctypes.byref(time), 8, None)
        return time.value

    def readHealth(self):
        base_address = ctypes.c_long(0x00883DB4)
        self.kernel.ReadProcessMemory(int(self.process_handle), base_address.value, ctypes.byref(base_address), 4, None)
        for i in self.health_offset:
            self.kernel.ReadProcessMemory(int(self.process_handle), base_address.value + i, ctypes.byref(base_address),
                                          4, None)

        health = ctypes.c_double()
        self.kernel.ReadProcessMemory(int(self.process_handle), base_address, ctypes.byref(health), 8, None)
        return health.value

    # flash cooldown
    def readFlash(self):
        base_address = ctypes.c_long(0x00883DB4)
        self.kernel.ReadProcessMemory(int(self.process_handle), base_address.value, ctypes.byref(base_address), 4, None)
        for i in self.flash_offset:
            self.kernel.ReadProcessMemory(int(self.process_handle), base_address.value + i, ctypes.byref(base_address),
                                          4, None)

        flash = ctypes.c_double()
        self.kernel.ReadProcessMemory(int(self.process_handle), base_address, ctypes.byref(flash), 8, None)
        if flash.value == 720.0:
            return 0.0
        return flash.value

    def readPosition(self):
        base_address = ctypes.c_long(0x00883DB4)
        self.kernel.ReadProcessMemory(int(self.process_handle), base_address.value, ctypes.byref(base_address), 4, None)
        self.kernel.ReadProcessMemory(int(self.process_handle), base_address.value + 0x38, ctypes.byref(base_address),
                                          4, None)

        position_x = ctypes.c_int()
        self.kernel.ReadProcessMemory(int(self.process_handle), base_address.value+0xF0, ctypes.byref(position_x), 4, None)

        base_address = ctypes.c_long(0x00883DB4)
        self.kernel.ReadProcessMemory(int(self.process_handle), base_address.value, ctypes.byref(base_address), 4, None)
        self.kernel.ReadProcessMemory(int(self.process_handle), base_address.value + 0x38, ctypes.byref(base_address),
                                      4, None)

        position_y = ctypes.c_int()
        self.kernel.ReadProcessMemory(int(self.process_handle), base_address.value+0xEC, ctypes.byref(position_y), 4, None)
        return [position_x.value,position_y.value]




    # time, health, flash time
    def getAllData(self):
        return [self.readTime(), self.readHealth(), self.readFlash()]



'''''
start = time.perf_counter()
count=0
while True:
    data.grabScreen()
    end=time.perf_counter()
    count+=1
    if(end-start>1):
        print(count)
        break
    '''''
