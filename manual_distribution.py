# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 12:48:16 2022

@author: zhangq
"""

import numpy as np
import matplotlib.pyplot as plt
from imageio import imread, imwrite
import cv2 as cv


def fixedHist(x, binWidth):
    binEdges = np.arange(np.min(x) - binWidth / 2.0, np.max(x) + binWidth / 2.0, binWidth)
    if binEdges[-1] < np.max(x): 
        binEdges = np.append(binEdges, np.max(x) + binWidth / 2.0)
    hy, binEdges = np.histogram(x, binEdges, density = True)
    hx = (binEdges[:-1] + binEdges[1:]) / 2
    return hx, hy

# fn = 'zetica_data\\raw\\crib_0000000_0000000_425_866.png'
# im = imread(fn)
# im0 = im[20:-20, 20:637]
# plt.imshow(im0, cmap = 'gray', interpolation = 'none')
# raise SystemExit

img = imread('test.png')
M, N, K = np.shape(img)
edgeImg = img[:, : , 0]
contours, hierarchy = cv.findContours(edgeImg, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

cv.drawContours(img, contours[:200], -1, (255, 0, 0), 1)

plt.imshow(img[:, :, :3], interpolation = 'none')

raise SystemExit
meansizes = []
for contour in contours:
    x, y, w, h = cv.boundingRect(contour)
    meansizes.append((w + h) / 2)
hx, hy = fixedHist(meansizes, 1)
plt.plot(hx, hy)
plt.grid(True)