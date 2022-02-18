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
leftMeans = []
rightMeans = []
coefs = []
for i, fn in enumerate(fns):
    if i % 10 == 0:
        print('%d out of %d' % (i, len(fns)))
    '''Read in the image'''
    im_raw = imread(fn)
    im0 = im_raw[edgeCrop:-edgeCrop, edgeCrop:637]
    im1 = im_raw[edgeCrop:-edgeCrop, 637:-edgeCrop]
    M, N = np.shape(im0)
    mask = np.array([[True] * N] * M)
    bkg, mse, cofs = test_zetica.quadratic_surfacefit_mask(im0, mask)
    coefs.append(cofs)
    plt.subplot(2,2,1)
    plt.imshow(im0, interpolation = 'none', cmap = 'gray')
    plt.title('Original')
    plt.subplot(2,2,2)
    plt.imshow(bkg, interpolation = 'none')
    plt.title('Fitted background')
    plt.subplot(2,2,3)
    plt.imshow(im0 - bkg, interpolation = 'none', cmap = 'gray')
    plt.title('Background corrected')
    raise SystemExit
    # plt.subplot(3,4,i + 1)
    # plt.imshow(bkg, interpolation = 'none')
    # plt.title('%s(left)' % fn[fn.rfind('\\') + 1:-4])
    
# plt.gcf().set_size_inches(16, 12)
# plt.gcf().tight_layout()
# plt.savefig('polyfit_background_left.png', dpi = 300)

coefs = np.array(coefs)
titles = ['a', 'b', 'c', 'd', 'e', 'f']
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.plot(coefs[:, i])
    if i == 0:
        plt.title(r'a$(ax^2+by^2+cxy+dx+ey+f)$')
    else:
        plt.title(titles[i])
    plt.grid(True)