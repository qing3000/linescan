# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 15:57:04 2022

@author: zhangq
"""
import numpy as np
import matplotlib.pyplot as plt
from imageio import imread, imwrite
import test_zetica
import glob

fns = glob.glob('zetica_data\\raw\\*.png')
edgeCrop = 20
maxscale = 6
leftMeans = []
rightMeans = []
for i, fn in enumerate(fns):
    if i % 10 == 0:
        print('%d out of %d' % (i, len(fns)))
    '''Read in the image'''
    im_raw = imread(fn)
    im0 = im_raw[edgeCrop:-edgeCrop, edgeCrop:637]
    im1 = im_raw[edgeCrop:-edgeCrop, 637:-edgeCrop]
    
    wavelengths, psd = test_zetica.calculate_PSD(im0.astype('double'), maxscale)
    leftMeans.append(np.sum(psd * wavelengths))
    wavelengths, psd = test_zetica.calculate_PSD(im1.astype('double'), maxscale)
    rightMeans.append(np.sum(psd * wavelengths))
    # plt.plot(wavelengths, psd)
    # plt.grid(True)
    # plt.title(meansize)
np.savetxt('sizemeans.csv', np.array([leftMeans, rightMeans]).T, fmt = '%f', delimiter = ',')
plt.plot(leftMeans, label = 'Left half crib')
plt.plot(rightMeans, label = 'Right half crib')
plt.grid(True)
           
       