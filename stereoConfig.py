# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 20:17:47 2022

@author: lenovo
"""

# -*- coding: utf-8 -*-
# @Time : 2022/3/25 16:06
# @Author : Zhang Jun
# @File : stereoConfig.py
# @Software: PyCharm
import numpy as np


####################仅仅是一个示例###################################


# 双目相机参数
class stereoCamera(object):
    def __init__(self):
        # 左相机内参(已改)
        self.cam_matrix_left = np.array([[7.372711487536764e+02, 0,6.202947387364632e+02],
                                         [0., 7.357936742279408e+02, 3.400548535216879e+02],
                                         [0., 0., 1.]])
        # 右相机内参（已改）
        self.cam_matrix_right = np.array([[7.342419838786009e+02, 0,  6.088782690997340e+02],
                                          [0., 7.329442497556776e+02,3.374334410011336e+02],
                                          [0., 0., 1.]])

        # 左右相机畸变系数:[k1, k2, p1, p2, k3]（已换）
        self.distortion_l = np.array([[0.089937521302425, -0.008605012820637, -1.422054811710335e-04, 0.002784607837870,-0.196354738921342]])
        self.distortion_r = np.array([[0.076447346943265, 0.088843534383102,1.441249730313277e-04, 0.002824182354917,-0.365044917892326]])

        # 旋转矩阵（已改）
        self.R = np.array([[0.999958460819669, 0.001988053535107,0.008895182870510],
                           [ -0.001967217636689, 0.999995302551878, -0.002350516740654],
                           [-0.008899814038965,0.002332920341491, 0.999957674500652]])

        # 平移矩阵（已改）
        self.T = np.array([[-59.779882896526246], [-0.121031263411344], [-0.002428898329554]])

        # 主点列坐标的差
        self.doffs = 0.0

        # 指示上述内外参是否为经过立体校正后的结果
        self.isRectified = False

    # def setMiddleBurryParams(self):
    #     self.cam_matrix_left = np.array([[3997.684, 0, 225.0],
    #                                      [0., 3997.684, 187.5],
    #                                      [0., 0., 1.]])
    #     self.cam_matrix_right = np.array([[3997.684, 0, 225.0],
    #                                       [0., 3997.684, 187.5],
    #                                       [0., 0., 1.]])
    #     self.distortion_l = np.zeros(shape=(5, 1), dtype=np.float64)
    #     self.distortion_r = np.zeros(shape=(5, 1), dtype=np.float64)
    #     self.R = np.identity(3, dtype=np.float64)
    #     self.T = np.array([[-193.001], [0.0], [0.0]])
    #     self.doffs = 131.111
    #     self.isRectified = True
    def setMiddleBurryParams(self):
        self.cam_matrix_left = np.array([[7.372711487536764e+02, 0,6.202947387364632e+02],
                                         [0., 7.357936742279408e+02, 3.400548535216879e+02],
                                         [0., 0., 1.]])
        self.cam_matrix_right = np.array([[7.342419838786009e+02, 0,  6.088782690997340e+02],
                                          [0., 7.329442497556776e+02,3.374334410011336e+02],
                                          [0., 0., 1.]])
        self.distortion_l = np.zeros(shape=(5, 1), dtype=np.float64)
        self.distortion_r = np.zeros(shape=(5, 1), dtype=np.float64)
        self.R = np.identity(3, dtype=np.float64)
        self.T = np.array([[-59.779882896526246], [-0.121031263411344], [-0.002428898329554]])
        self.doffs = 131.111
        self.isRectified = True
