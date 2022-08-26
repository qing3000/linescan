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
import tqdm

def calculate_mean_size(fn):
    maxscale = 80
    im_raw = imread(fn)
    im = im_raw[25:-30, :-20]
    
    # im = enhance_image_quality(im)
    
    wavelengths, psd = calculate_PSD(im.astype('double'), maxscale)
    meansize = np.sum(psd * wavelengths)
    return meansize

def image_height(fn):
    return len(imread(fn))

def enhance_im_quality(fn):
    im0 = imread(fn)
    im1 = enhance_image_quality(im0)
    return im1

if __name__ == '__main__':
    fns = glob('..\\mud\\Output\\Cribs\\Run_354\\*.png')

    mp.freeze_support()
    pool = mp.Pool(mp.cpu_count())

    heights = []
    for height in tqdm.tqdm(pool.imap(image_height, fns), total = len(fns)):
        heights.append(height)
    heights = np.array(heights)
    fns = np.array(fns)
    short_fns = fns[np.logical_and(heights > 0, heights < 500)]
    
    meansizes = []
    for meansize in tqdm.tqdm(pool.imap(calculate_mean_size, short_fns), total = len(short_fns)):
        meansizes.append(meansize)
    pool.close()
    
    f = open('meansizes_full.csv', 'w')
    f.write('Filename,Meansize(mm)\n')
    for short_fn, meansize in zip(short_fns, meansizes):
        ss = short_fn[short_fn.rfind('\\') + 1:]
        f.write('%s,%.2f\n' % (ss, meansize))
    f.close()
    
    plt.plot(meansizes)
    plt.grid(True)
    plt.show()