# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 15:24:31 2022

@author: lenovo
"""

#import sys
import cv2
#import numpy as np
#import stereoConfig
#import open3d as o3d
import time
import stereo
AUTO = True  # 自动拍照，或手动按s键拍照
INTERVAL = 8 # 自动拍照间隔
 
cv2.namedWindow("left")
cv2.namedWindow("right")
camera = cv2.VideoCapture(0) #读帧
 
# 设置分辨率左右摄像机同一频率，同一设备ID；左右摄像机总分辨率2560x720；分割为两个1280x720
camera.set(cv2.CAP_PROP_FRAME_WIDTH,2560)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
 
counter = 0
utc = time.time()
#folder = "./capture/" # 拍照文件目录
folder1 = "./left/"
folder2 = "./right/"

def shot(frame,path):
    global counter
    #path = folder + pos + "_" + str(counter) + ".jpg"
 
    cv2.imwrite(path, frame)
    print("snapshot saved into: " + path)
 
while True:
    ret, frame = camera.read() #frame是每一帧的图像，是一个三维矩阵
    #cv2.imshow("final",frame)
    #print("ret:",ret)
    # 裁剪坐标为[y0:y1, x0:x1]    HEIGHT * WIDTH
    left_frame = frame[0:720, 0:1280]
    right_frame = frame[0:720, 1280:2560]
 
    cv2.imshow("left", left_frame)
    cv2.imshow("right", right_frame)
 
    now = time.time()
    if AUTO and now - utc >= INTERVAL:
        path1 = folder1 + "left" +".jpg"   # "_" + str(counter) +
        path2 = folder2 + "right"+ ".jpg"  # + "_" + str(counter)
        shot(left_frame,path1)
        shot(right_frame,path2)
        #counter += 1
        utc = now
        minline=stereo.runout(path1,path2)
        print("目前最近距离%2f mm" % minline)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord("s"):
        shot("left", left_frame,folder1)
        shot("right", right_frame,folder1)
        counter += 1
camera.release()
cv2.destroyWindow("left")
cv2.destroyWindow("right")