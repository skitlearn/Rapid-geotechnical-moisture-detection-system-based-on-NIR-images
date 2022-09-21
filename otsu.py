#学习课程
#python
#学习时间: 2022-08-15  14:49


# import numpy as np
# import cv2
# import math
# import time as dt
# import matplotlib.pyplot as plt
#
# image = cv2.imread('../images/ce_7.jpg', 0)
# rows, cols = image.shape[:2]
# gray_hist = np.zeros([256], np.uint64)
# for i in range(rows):
#     for j in range(cols):
#         gray_hist[image[i][j]] += 1
# uniformGrayHist = gray_hist / float(rows * cols)
# # 计算零阶累积距和一阶累积距
# zeroCumuMomnet = np.zeros(256, np.float32)
# oneCumuMomnet = np.zeros(256, np.float32)
# for k in range(256):
#     if k == 0:
#         zeroCumuMomnet[k] = uniformGrayHist[0]
#         oneCumuMomnet[k] = (k) * uniformGrayHist[0]
#     else:
#         zeroCumuMomnet[k] = zeroCumuMomnet[k - 1] + uniformGrayHist[k]
#         oneCumuMomnet[k] = oneCumuMomnet[k - 1] + k * uniformGrayHist[k]
# # 计算类间方差
# variance = np.zeros(256, np.float32)
# for k in range(255):
#     if zeroCumuMomnet[k] == 0 or zeroCumuMomnet[k] == 1:
#         variance[k] = 0
#     else:
#         variance[k] = math.pow(oneCumuMomnet[255] * zeroCumuMomnet[k] - oneCumuMomnet[k], 2) / (
#                     zeroCumuMomnet[k] * (1.0 - zeroCumuMomnet[k]))
# # 找到阈值
# threshLoc = np.where(variance[0:255] == np.max(variance[0:255]))
# thresh = threshLoc[0][0]
# # 阈值处理
# threshold = np.copy(image)
# threshold[threshold > thresh] = 255
# threshold[threshold <= thresh] = 0
# cv2.imshow('raw', image)
# cv2.imshow("test", threshold)
# cv2.waitKey(0)





# import cv2
# import matplotlib.pyplot as plt
# 读图
# img = cv2.imread('../images/ce_6.jpg', 0)
# 转换成灰度图
# img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# 显示灰度图
# cv2.imshow('gray_img',img)
# cv2.waitKey(0)
# 获取直方图，由于灰度图img2是二维数组，需转换成一维数组
# plt.hist(img.ravel(),256)
# 显示直方图
# plt.show()
# cv2.waitKey(0)


#
# 单阈值OTSU分割
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# image = cv2.imread("../images/ce_7.jpg")
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# plt.subplot(131), plt.imshow(image, "gray")
# plt.title("source image"), plt.xticks([]), plt.yticks([])
# plt.subplot(132), plt.hist(image.ravel(), 256)
# plt.title("Histogram"), plt.xticks([]), plt.yticks([])
# # ret1, th1 = cv2.threshold(gray,170,255,cv2.THRESH_BINARY)
# ret1, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  #方法选择为THRESH_OTSU
# # ret1, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# plt.subplot(133), plt.imshow(th1, "gray")
# plt.title("OTSU,threshold is " + str(ret1)), plt.xticks([]), plt.yticks([])
# plt.show()




# import cv2
# import numpy as np
# from numpy.core.fromnumeric import shape
# from PIL import Image
# #plotly
# import plotly as py
# import plotly.graph_objs as go
#
# img = cv2.imread('road.jpg',0)
# img1 = np.ravel(img)
# print('type', type(img1), img1.shape, img1)
# pyplt = py.offline.plot
# data = [go.Histogram(x=img1, histnorm = 'probability')]
# ret, th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# print('ret', ret, cv2.THRESH_BINARY_INV, cv2.THRESH_OTSU)
# pyplt(data)
# self.show_img(img, 'road')
# self.show_img(th2, 'th2')


# import cv2
# import numpy as np
# img = cv2.imread('../images/ce_6.jpg',0)
# blur = cv2.GaussianBlur(img,(5,5),0)
# # find normalized_histogram, and its cumulative distribution function
# hist = cv2.calcHist([blur],[0],None,[256],[0,256])
# hist_norm = hist.ravel()/hist.max()
# Q = hist_norm.cumsum()
# bins = np.arange(256)
# print(bins)
# fn_min = np.inf
# thresh = -1
# for i in range(1,256):
#     p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
#     q1,q2 = Q[i],Q[255]-Q[i] # cum sum of classes
#     b1,b2 = np.hsplit(bins,[i]) # weights
#
#     # finding means and variances
#     m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
#     v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2
#
#     # calculates the minimization function
#     fn = v1*q1 + v2*q2
#     if fn < fn_min:
#         fn_min = fn
#         thresh = i
# # find otsu's threshold value with OpenCV function
# ret, otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# print(thresh,ret)




# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# img = cv2.imread('../images/ce_6.jpg',0)
# img = cv2.GaussianBlur(img, (5, 5), 0)
#
# ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
#             cv2.THRESH_BINARY,11,2)
# th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#             cv2.THRESH_BINARY,11,2)
#
# titles = ['Original Image', 'Global Thresholding (v = 127)',
#             'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
# images = [img, th1, th2, th3]
#
# for i in range(4):
#     plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
# plt.show()




import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import util
from skimage.util import random_noise







img = cv2.imread('ce_1234.jpg',0)
# noisy = random_noise(img, mode='gaussian', mean=0, var=0.01 ** 2)
# cv2.imwrite('ce_1234.jpg',noisy*255)

# global thresholding
ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

# Otsu's thresholding
ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(img,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# plot all the images and their histograms
images = [img, 0, th1,
          img, 0, th2,
          blur, 0, th3]
titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
          'Original Noisy Image','Histogram',"Otsu's Thresholding",
          'Gaussian filtered Image','Histogram',"Otsu's Thresholding "+ str(ret3)]

for i in range(3):
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
plt.show()



