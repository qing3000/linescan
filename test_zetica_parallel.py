# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 16:17:00 2022

@author: zhangq
"""
import numpy as np
import matplotlib.pyplot as plt

from test_zetica import calculate_PSD, enhance_image_quality
from imageio import imread
from glob import glob


import multiprocessing as mp

def calculate_mean_size(fn):
    maxscale = 80
    im_raw = imread(fn)
    im = im_raw[25:-30, :-20]
    
    im = enhance_image_quality(im)
    
    wavelengths, psd = calculate_PSD(im.astype('double'), maxscale)
    meansize = np.sum(psd * wavelengths)
    return meansize

if __name__ == '__main__':
    mp.freeze_support()
    pool = mp.Pool(mp.cpu_count())
    fns = glob('zetica_data\\cribimages\\*.png')
    meansizes = pool.map(calculate_mean_size, [fn for fn in fns])
    pool.close()
    
    plt.plot(meansizes)
    plt.grid(True)