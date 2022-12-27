import cv2 as cv
import detect
#import zhangai   障碍模块
# 语音播报模块
import pyttsx3
import time
import stereo



def say0(text):
    engine.say(text)
    engine.runAndWait()
    print('检测到目标' + text + '人脸')

def say1():
    engine.say('重复')
    engine.runAndWait()
    print('重复检测到目标人脸')

def shut():
    engine.say('失败')
    engine.runAndWait()
    print('没有成功检测到目标人脸')

# 摄像头
#cap = cv.VideoCapture(0)
AUTO = True  # 自动拍照，或手动按s键拍照
INTERVAL = 8  # 自动拍照间隔

cv.namedWindow("left")
cv.namedWindow("right")
camera = cv.VideoCapture(0)  # 读帧

# 设置分辨率左右摄像机同一频率，同一设备ID；左右摄像机总分辨率2560x720；分割为两个1280x720
camera.set(cv.CAP_PROP_FRAME_WIDTH, 2560)
camera.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

counter = 0
utc = time.time()
# folder = "./capture/" # 拍照文件目录
folder1 = "./left/"
folder2 = "./right/"


def shot(frame, path):
    global counter
    # path = folder + pos + "_" + str(counter) + ".jpg"

    cv.imwrite(path, frame)
    print("snapshot saved into: " + path)

flag = 0
#num = 0
engine = pyttsx3.init()
minline=0

while True:
    ret, frame = camera.read()  # frame是每一帧的图像，是一个三维矩阵
    # print("ret:",ret)
    # 裁剪坐标为[y0:y1, x0:x1]    HEIGHT * WIDTH
    left_frame = frame[0:720, 0:1280]
    right_frame = frame[0:720, 1280:2560]
    #cv.imshow("frame", frame)
    cv.imshow("left", left_frame)
    cv.imshow("right", right_frame)

    now = time.time()
    if AUTO and now - utc >= INTERVAL:
        path1 = folder1 + "left" + ".jpg"  # "_" + str(counter) +
        path2 = folder2 + "right" + ".jpg"  # + "_" + str(counter)
        shot(left_frame, path1)
        shot(right_frame, path2)
        # counter += 1
        utc = now
        minline = stereo.runout(path1, path2)

        print("目前最近距离 %2f mm" % minline)
    key = cv.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord("s"):
        shot("left", left_frame, folder1)
        shot("right", right_frame, folder1)
        counter += 1

    engine.say("目前的障碍物距离是"+str(minline)+"mm")

    #ret_flag, Vshow = camera.read()  # 得到每帧图像
    #cv.imshow("Capture_Test", Vshow)  # 显示图像

    if(flag == 0):
        text = detect.action(ret, frame)
        text2 = text
        if text != 'unknown':
            say0(text)
            flag = 1
        else:
            shut()
    else:
        text = detect.action(ret, frame)
        if text != 'unknown' and text != text2:
            say0(text)
            text2 = text
        if text == text2:
            say1()
        else:
            shut()
camera.release()
cv2.destroyWindow("left")
cv2.destroyWindow("right")




# while (camera.isOpened()):  # 检测是否在开启状态
#
#     ret_flag, Vshow = camera.read()  # 得到每帧图像
#     #cv.imshow("Capture_Test", Vshow)  # 显示图像
#
#     if(flag == 0):
#         text = detect.action(ret_flag, Vshow)
#         text2 = text
#         if text != 'unknown':
#             say0(text)
#             flag = 1
#         else:
#             shut()
#     else:
#         text = detect.action(ret_flag, Vshow)
#         if text != 'unknown' and text != text2:
#             say0(text)
#             text2 = text
#         if text == text2:
#             say1()
#         else:
#             shut()