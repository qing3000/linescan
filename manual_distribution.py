# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 12:48:16 2022

@author: zhangq
"""

import numpy as np
import matplotlib.pyplot as plt
from imageio import imread, imwrite
import cv2 as cv
import test_zetica as tt

def fixedHist(x, binWidth):
    binEdges = np.arange(np.min(x) - binWidth / 2.0, np.max(x) + binWidth / 2.0, binWidth)
    if binEdges[-1] < np.max(x): 
        binEdges = np.append(binEdges, np.max(x) + binWidth / 2.0)
    hy, binEdges = np.histogram(x, binEdges, density = True)
    hx = (binEdges[:-1] + binEdges[1:]) / 2
    return hx, hy

fn = 'zetica_data\\raw\\crib_0000000_0000000_425_866.png'
im = imread(fn)
im0 = im[20:-20, 20:637]
#plt.imshow(im0, cmap = 'gray', interpolation = 'none')
row = im0[100].astype('float')
d1 = np.diff(row)
d2 = np.diff(row, 2)
plt.subplot(3,1,1)
plt.plot(row)
plt.grid(True)
plt.subplot(3,1,2)
plt.plot(d1)
plt.grid(True)
plt.subplot(3,1,3)
plt.plot(d2)
plt.grid(True)

raise SystemExit

img = imread('test.png')
M, N, K = np.shape(img)
edgeImg = img[:, : , 0]
erodeKernel = np.ones((3, 3), np.uint8)
erodedImg = cv.erode(edgeImg, erodeKernel)
contours, hierarchy = cv.findContours(erodedImg, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

boxes = []
meansizes = []
for contour in contours:
    if cv.contourArea(contour) > 10:
        rect = cv.minAreaRect(contour)
        boxes.append(np.int0(cv.boxPoints(rect)))
        meansizes.append(np.mean(rect[1]) + 2)
    
cv.drawContours(img, contours, -1, (255, 0, 0), 1)
cv.drawContours(img, boxes, -1, (0, 0, 255), 1)
hx, hy = fixedHist(meansizes, 2)

fn = 'zetica_data\\raw\\crib_0000000_0000000_425_866.png'
im0 = plt.imread(fn)
im = im0[20:-20, 20:637]
wavelengths, fullPSD = tt.calculate_PSD(im, 80)

plt.subplot(2,2,1)
plt.imshow(img[:, :, :3], interpolation = 'none')
plt.subplot(2,2,2)
plt.imshow(erodedImg, interpolation = 'none', cmap = 'gray')
plt.subplot(2,2,3)
plt.plot(hx, hy)
plt.plot(wavelengths, fullPSD)
plt.grid(True)
