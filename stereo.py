# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 20:18:33 2022

@author: lenovo
"""

# -*- coding: utf-8 -*-
# @Time : 2022/3/25 16:05
# @Author : Zhang Jun
# @File : stereo.py
# @Software: PyCharm
# -*- coding: utf-8 -*-
import sys
import cv2
import numpy as np
import stereoConfig
#import open3d as o3d
import time

# 预处理
def preprocess(img1, img2):
    # 彩色图->灰度图
    if (img1.ndim == 3):
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 通过OpenCV加载的图像通道顺序是BGR
    if (img2.ndim == 3):
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 直方图均衡
    img1 = cv2.equalizeHist(img1)
    img2 = cv2.equalizeHist(img2)

    return img1, img2


# 消除畸变
def undistortion(image, camera_matrix, dist_coeff):
    undistortion_image = cv2.undistort(image, camera_matrix, dist_coeff)

    return undistortion_image


# 获取畸变校正和立体校正的映射变换矩阵、重投影矩阵
# @param：config是一个类，存储着双目标定的参数:config = stereoconfig.stereoCamera()
def getRectifyTransform(height, width, config):
    # 读取内参和外参
    left_K = config.cam_matrix_left
    right_K = config.cam_matrix_right
    left_distortion = config.distortion_l
    right_distortion = config.distortion_r
    R = config.R
    T = config.T

    # 计算校正变换
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(left_K, left_distortion, right_K, right_distortion,
                                                      (width, height), R, T, alpha=0)

    map1x, map1y = cv2.initUndistortRectifyMap(left_K, left_distortion, R1, P1, (width, height), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(right_K, right_distortion, R2, P2, (width, height), cv2.CV_32FC1)

    return map1x, map1y, map2x, map2y, Q


# 畸变校正和立体校正
def rectifyImage(image1, image2, map1x, map1y, map2x, map2y):
    rectifyed_img1 = cv2.remap(image1, map1x, map1y, cv2.INTER_AREA)
    rectifyed_img2 = cv2.remap(image2, map2x, map2y, cv2.INTER_AREA)

    return rectifyed_img1, rectifyed_img2


# 立体校正检验----画线
def draw_line(image1, image2):
    # 建立输出图像
    height = max(image1.shape[0], image2.shape[0])
    width = image1.shape[1] + image2.shape[1]

    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:image1.shape[0], 0:image1.shape[1]] = image1
    output[0:image2.shape[0], image1.shape[1]:] = image2

    # 绘制等间距平行线
    line_interval = 20  # 直线间隔：50
    for k in range(height // line_interval):
        cv2.line(output, (0, line_interval * (k + 1)), (2 * width, line_interval * (k + 1)), (0, 255, 0), thickness=2,
                 lineType=cv2.LINE_AA)

    return output


# 视差计算
def stereoMatchSGBM(left_image, right_image, down_scale=False):
    # SGBM匹配参数设置
    if left_image.ndim == 2:
        img_channels = 1
    else:
        img_channels = 3
    blockSize = 3
    paraml = {'minDisparity': 2,    #此值对重建影响最大，不超过10最好，但大于0
              'numDisparities': 64, #影响三维重建的厚度
              'blockSize': blockSize,
              'P1': 8 * img_channels * blockSize ** 2,
              'P2': 32 * img_channels * blockSize ** 2,
              'disp12MaxDiff': -1,
              'preFilterCap': 1,
              'uniquenessRatio': 12, #5-15任意
              'speckleWindowSize': 0, #设成0或者50-100
              'speckleRange': 2,  #统常1或者2就够用，但调也没有什么影响
              'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
              }

    # 构建SGBM对象
    left_matcher = cv2.StereoSGBM_create(**paraml)
    paramr = paraml
    paramr['minDisparity'] = -paraml['numDisparities']
    right_matcher = cv2.StereoSGBM_create(**paramr)

    # 计算视差图
    size = (left_image.shape[1], left_image.shape[0])
    if down_scale == False:
        disparity_left = left_matcher.compute(left_image, right_image)
        disparity_right = right_matcher.compute(right_image, left_image)
    else:
        left_image_down = cv2.pyrDown(left_image) #从高分辨率的大尺寸图像逐次向下采样得到一系列图像，构建一个金字塔
        right_image_down = cv2.pyrDown(right_image)
        factor = left_image.shape[1] / left_image_down.shape[1]

        disparity_left_half = left_matcher.compute(left_image_down, right_image_down) #创建的sgbm对象里的函数，获取视差图
        disparity_right_half = right_matcher.compute(right_image_down, left_image_down)
        disparity_left = cv2.resize(disparity_left_half, size, interpolation=cv2.INTER_AREA)
        disparity_right = cv2.resize(disparity_right_half, size, interpolation=cv2.INTER_AREA)
        disparity_left = factor * disparity_left
        disparity_right = factor * disparity_right

    # 真实视差（因为SGBM算法得到的视差是×16的）
    trueDisp_left = disparity_left.astype(np.float32) / 16.
    trueDisp_right = disparity_right.astype(np.float32) / 16.

    return trueDisp_left, trueDisp_right


def getDepthMapWithQ(disparityMap: np.ndarray, Q: np.ndarray) -> np.ndarray:
    points_3d = cv2.reprojectImageTo3D(disparityMap, Q)
    depthMap = points_3d[:, :, 2]
    reset_index = np.where(np.logical_or(depthMap < 0.0, depthMap > 65535.0))
    depthMap[reset_index] = 0

    return depthMap.astype(np.float32)


def getDepthMapWithConfig(disparityMap: np.ndarray, config: stereoConfig.stereoCamera) -> np.ndarray:
    fb = config.cam_matrix_left[0, 0] * (-config.T[0])
    doffs = config.doffs
    depthMap = np.divide(fb, disparityMap + doffs)
    reset_index = np.where(np.logical_or(depthMap < 0.0, depthMap > 65535.0))
    depthMap[reset_index] = 0
    reset_index2 = np.where(disparityMap < 0.0)
    depthMap[reset_index2] = 0
    return depthMap.astype(np.float32)

def runout(iml_path,imr_path):
    iml = cv2.imread(iml_path, 1)  # 左图
    imr = cv2.imread(imr_path, 1)  # 右图
    if (iml is None) or (imr is None):
        print("Error: Images are empty, please check your image's path!")
        sys.exit(0)
    height, width = iml.shape[0:2]

    # 读取相机内参和外参
    # 使用之前先将标定得到的内外参数填写到stereoconfig.py中的StereoCamera类中
    config = stereoConfig.stereoCamera()
    config.setMiddleBurryParams()
    #print(config.cam_matrix_left)

    # 立体校正
    map1x, map1y, map2x, map2y, Q = getRectifyTransform(height, width, config)  # 获取用于畸变校正和立体校正的映射矩阵以及用于计算像素空间坐标的重投影矩阵
    iml_rectified, imr_rectified = rectifyImage(iml, imr, map1x, map1y, map2x, map2y)
    #print(Q)

    # 绘制等间距平行线，检查立体校正的效果
    line = draw_line(iml_rectified, imr_rectified)
    cv2.imwrite('check_rectification.png', line)

    # 立体匹配
    iml_, imr_ = preprocess(iml, imr)  # 预处理，一般可以削弱光照不均的影响，不做也可以
    disp, _ = stereoMatchSGBM(iml, imr)  # , True这里传入的是未经立体校正的图像，因为我们使用的middleburry图片已经是校正过的了
    cv2.imwrite('disaprity.png', disp)

    # 计算深度图
    #depthMap = getDepthMapWithQ(disp, Q)
    depthMap = getDepthMapWithConfig(disp, config)
    minDepth = np.min(depthMap) #深度图的最小值
    maxDepth = np.max(depthMap)
    #print(depthMap)
    #print(np.shape(depthMap))  (720,1280)
    #print(minDepth, maxDepth)
    depthmap_average=np.ones((72,128))*0
    
    #print(sum(map(sum,depthMap[0:9,0:9]))/100)
    for i in range(0,72):
        for j in range(0,128):
            depthmap_average[i][j]=sum(map(sum,depthMap[i*10:(i+1)*10-1,j*10:(j+1)*10-1]))/100
            #print(depthmap_average[i][j])
    print('目前最近的障碍物距离为 %2f mm' % np.min(depthmap_average))
    
    return np.min(depthmap_average)

# if __name__ == '__main__':
#     # 读取MiddleBurry数据集的图片
#     #自己拍摄的图片集
#     iml = cv2.imread('camer2/left/left_4.jpg', 1)  # 左图
#     imr = cv2.imread('camer2/right/right_4.jpg', 1)  # 右图
#     if (iml is None) or (imr is None):
#         print("Error: Images are empty, please check your image's path!")
#         sys.exit(0)
#     height, width = iml.shape[0:2]

#     # 读取相机内参和外参
#     # 使用之前先将标定得到的内外参数填写到stereoconfig.py中的StereoCamera类中
#     config = stereoConfig.stereoCamera()
#     config.setMiddleBurryParams()
#     #print(config.cam_matrix_left)

#     # 立体校正
#     map1x, map1y, map2x, map2y, Q = getRectifyTransform(height, width, config)  # 获取用于畸变校正和立体校正的映射矩阵以及用于计算像素空间坐标的重投影矩阵
#     iml_rectified, imr_rectified = rectifyImage(iml, imr, map1x, map1y, map2x, map2y)
#     #print(Q)

#     # 绘制等间距平行线，检查立体校正的效果
#     line = draw_line(iml_rectified, imr_rectified)
#     cv2.imwrite('check_rectification.png', line)

#     # 立体匹配
#     iml_, imr_ = preprocess(iml, imr)  # 预处理，一般可以削弱光照不均的影响，不做也可以
#     disp, _ = stereoMatchSGBM(iml, imr)  # , True这里传入的是未经立体校正的图像，因为我们使用的middleburry图片已经是校正过的了
#     cv2.imwrite('disaprity.png', disp)

#     # 计算深度图
#     #depthMap = getDepthMapWithQ(disp, Q)
#     depthMap = getDepthMapWithConfig(disp, config)
#     minDepth = np.min(depthMap) #深度图的最小值
#     maxDepth = np.max(depthMap)
#     #print(depthMap)
#     #print(np.shape(depthMap))  (720,1280)
#     #print(minDepth, maxDepth)
#     depthmap_average=np.ones((72,128))*0
    
#     #print(sum(map(sum,depthMap[0:9,0:9]))/100)
#     for i in range(0,72):
#         for j in range(0,128):
#             depthmap_average[i][j]=sum(map(sum,depthMap[i*10:(i+1)*10-1,j*10:(j+1)*10-1]))/100
#             #print(depthmap_average[i][j])
#     print('目前最近的障碍物距离为 %2f mm' % np.min(depthmap_average))
    
   
