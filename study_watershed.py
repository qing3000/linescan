# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 14:43:49 2022

@author: zhangq
"""

import numpy as np
from matplotlib import pyplot as plt

from skimage import data, segmentation, color
from skimage.future import graph
from skimage.io import imread
from skimage.filters import median
from skimage.morphology import disk, black_tophat, opening, closing, erosion, dilation, reconstruction
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage as ni
from skimage.color import rgb2gray

fn = 'pears.png'
I = rgb2gray(imread(fn))
se = disk(20)
Ie = erosion(I, se)
Iobr = reconstruction(Ie, I)

plt.plot(I[100], label = 'Marker')
plt.plot(Ie[100], label = 'Eroded')
plt.plot(Iobr[100], label = 'Reconstructed')
plt.legend(loc = 0)
plt.grid(True)
raise SystemExit
plt.subplot(1,2,1)
plt.imshow(Ie, interpolation = 'none', cmap = 'gray')
plt.subplot(1,2,2)
plt.imshow(Iobr, interpolation = 'none', cmap = 'gray')

raise SystemExit
#img = im[20:-20, 20:637]
img = im[20:200, 20:200]
img1 = median(img, disk(3))
img2 = black_tophat(img1, disk(5))

t1 = np.percentile(img2.flat, 60)
t2 = np.percentile(img2.flat, 70)

img3 = img2 < t2

distance = ni.distance_transform_edt(img3)
max_coords = peak_local_max(distance, labels = img2, footprint = np.ones((3, 3)))
local_maxima = np.zeros_like(img2, dtype = bool)
local_maxima[tuple(max_coords.T)] = True
markers = ni.label(local_maxima)[0]
labels = watershed(-distance, markers)

# labels = watershed(img3, ni.label(img3)[0])
# plt.imshow(labels, interpolation = 'none', cmap = 'gray')
# raise SystemExit

plt.subplot(3,3,1)
plt.imshow(img, interpolation = 'none', cmap = 'gray')
plt.title('Raw')
plt.colorbar()
plt.subplot(3,3,2)
plt.imshow(img1, interpolation = 'none', cmap = 'gray')
plt.title('Median filtered')
plt.colorbar()
plt.subplot(3,3,3)
plt.imshow(img2, interpolation = 'none', cmap = 'gray')
plt.title('bottom hat filtered')
plt.colorbar()
plt.subplot(3,3,4)
plt.imshow(img3)
plt.title('Thresholded')
plt.colorbar()
plt.subplot(3,3,5)
plt.imshow(distance)
plt.colorbar()
plt.title('Distance transform')
plt.subplot(3,3,6)
plt.imshow(markers)
plt.colorbar()
plt.title('Markers')
plt.subplot(3,3,7)
plt.imshow(labels)
plt.title('Labels')
plt.colorbar()
plt.subplot(3,3,8)
plt.imshow(labels)
#plt.imshow(img, interpolation = 'none', cmap = 'gray')
#plt.imshow(img2 < t2, interpolation = 'none', cmap = 'gray')
plt.colorbar()
# plt.subplot(2,2,3)
# plt.imshow(img2 > t2, interpolation = 'none', cmap = 'gray')
# plt.colorbar()
raise SystemExit


M, N = np.shape(img)
labels1 = np.reshape(np.arange(M * N), (M, N))

# labels1 = segmentation.slic(img, compactness=30, n_segments=400,
#                             start_label=1)
# out1 = color.label2rgb(labels1, img, kind='avg', bg_label=0)

g = graph.rag_mean_color(img, labels1, mode = 'similarity')
labels2 = graph.cut_normalized(labels1, g)
out2 = color.label2rgb(labels2, img, kind='avg', bg_label=0)

plt.subplot(2,2,1)
plt.imshow(img, interpolation = 'none', cmap = 'gray')
plt.subplot(2,2,2)
plt.imshow(out2, interpolation = 'none')
