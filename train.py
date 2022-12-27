# 导入模块
import cv2
import os
import sys
from PIL import Image
import numpy as np

# 摄像头
cap = cv2.VideoCapture(0)

flag = 1
num = 41


# 检测函数
def face_detect_demo():
    gary = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_detect = cv2.CascadeClassifier('D:/opencv/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml')
    face = face_detect.detectMultiScale(gary, 1.01, 10, 0, (100, 100), (300, 300))
    # 1.01是放大倍数，5是检测的次数，后两个坐标是限制人脸的范围
    for x, y, w, h in face:
        cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
    cv2.imshow('result', img)


def getImageAndLabels(path):
    # 存储人脸数据
    facesSamples = []
    # 存储人的姓名
    ids = []
    # 存储图片信息
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    face_detector = cv2.CascadeClassifier('D:/opencv/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml')
    # 打印数组imagePaths
    print('数据排列：', imagePaths)
    # 遍历列表中的图片
    for imagePath in imagePaths:
        # 打开图片,黑白化
        # PIL有九种模式：1(黑白，P(灰白,RGB,RGBA,CMYK,I,F,YCbCr
        PIL_img = Image.open(imagePath).convert('L')
        # 将图像转换为数组，以黑白深浅
        img_numpy = np.array(PIL_img, 'uint8')
        # 获取图片人脸特征
        faces = face_detector.detectMultiScale(img_numpy)
        # 获取每张图片的id和姓名
        id = int(os.path.split(imagePath)[1].split('.')[0])
        # 预防无面容照片
        for x, y, w, h in faces:
            ids.append(id)
            facesSamples.append(img_numpy[y:y + h, x:x + w])
        # 打印脸部特征和id
        print('id:', id)
    print('fs:', facesSamples[0:4])
    return facesSamples, ids

while (cap.isOpened()):  # 检测是否在开启状态
    ret_flag, Vshow = cap.read()  # 得到每帧图像
    cv2.imshow("Capture_Test", Vshow)  # 显示图像
    k = cv2.waitKey(1) & 0xFF  # 按键判断
    if k == ord('1'):  # 保存
        cv2.imwrite("./data/jm/" + str(num) + ".lzy.jpg", Vshow)
        print("success to save" + str(num) + ".lzy.jpg")
        print("-------------------")

        # 读取图片
        img = cv2.imread("./data/jm/" + str(num) + ".lzy.jpg")
        # 检测函数
        face_detect_demo()
        # 等待
        cv2.waitKey(0)

        num += 1

    elif k == ord(' '):  # 退出
        break
        # 释放摄像头
        cap.release()


if __name__ == '__main__':
    # 图片路径
    path = './data/jm/'
    # 获取图像数组和id标签数组和姓名
    faces, ids = getImageAndLabels(path)
    # 获取训练对象
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(ids))
    # 保存文件
    recognizer.write('trainer/trainer.yml')


# 释放内存
cv2.destroyAllWindows()
