# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 15:41:15 2022

@author: zhangq
"""
import numpy as np
import matplotlib.pyplot as plt
import test_zetica as tt
from imageio import imread, imwrite
import glob
def calculate_meansize(fn, left = True):
    maxscale = 80
    edgeCrop = 20
    '''Read in the image'''
    im_raw = imread(fn)
    im_left = im_raw[edgeCrop:-edgeCrop, edgeCrop:637]
    im_right = im_raw[edgeCrop:-edgeCrop, 637:-edgeCrop]
    
    if left == True:
        im = im_left
    else:
        im = im_right
    wavelengths, psd = tt.calculate_PSD(im.astype('double'), maxscale)
    meansize = np.sum(psd * wavelengths)
    return im, wavelengths, psd

fns = glob.glob('zetica_data\\raw\\*.png')
sizemeans, temp = np.loadtxt('sizemeans.csv', delimiter = ',', unpack = True)
indices = np.argsort(sizemeans)

for i, index in enumerate(indices[:4]):
    im, wavelengths, psd = calculate_meansize(fns[index], True)
    meansize = np.sum(psd * wavelengths)
    plt.subplot(4, 4, i + 1)
    plt.imshow(im, interpolation = 'none', cmap = 'gray')
    plt.title('Mean Size=%.1fpixels' % meansize)
    plt.ylim([430, 0])
    plt.subplot(2,1,2)
    plt.plot(wavelengths, psd, 'b', label = 'Small particles')

for i, index in enumerate(indices[-4:]):
    im, wavelengths, psd = calculate_meansize(fns[index], True)
    meansize = np.sum(psd * wavelengths)
    plt.subplot(4, 4, i + 5)
    plt.imshow(im, interpolation = 'none', cmap = 'gray')
    plt.title('Mean Size=%.1fpixels' % meansize)
    plt.ylim([430, 0])
    plt.subplot(2,1,2)
    plt.plot(wavelengths, psd, 'r', label = 'Large particles')
plt.subplot(2,1,2)
plt.title('Size distribution')
plt.xlabel('Size (pixels)')
plt.legend(loc = 0)
plt.grid(True)

raise SystemExit
im2, wavelengths2, psd2 = calculate_meansize(fns[871], True)
meansize1 = np.sum(psd1 * wavelengths1)
meansize2 = np.sum(psd2 * wavelengths2)

plt.subplot(2,2,1)
plt.imshow(im1, interpolation = 'none', cmap = 'gray')
plt.subplot(2,2,2)
plt.imshow(im2, interpolation = 'none', cmap = 'gray')
plt.subplot(2,1,2)
plt.plot(wavelengths1, psd1, label = 'Mean size = %.1fpixels' %  meansize1)
plt.plot(wavelengths2, psd2, label = 'Mean size = %.1fpixels' %  meansize1)
plt.xlabel('Wavelength (pixels)')
plt.ylabel('PSD')
plt.grid(True)