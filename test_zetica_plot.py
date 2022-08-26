# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 16:17:00 2022

@author: zhangq
"""
import numpy as np
import matplotlib.pyplot as plt

from imageio import imread


if __name__ == '__main__':
    fpath = '..\\mud\\Output\\Cribs\\Run_354\\'
    range_starts = [80, 400, 540, 650, 830]
    for range_start in range_starts:
        range_end = range_start + 9
        fns, meansizes = np.genfromtxt('meansizes.csv', dtype = str, delimiter = ',', unpack = True, skip_header = 1)
        meansizes = list(map(float, meansizes))
        plt.figure(figsize = (16, 12))
        plt.subplot(4, 1, 1)
        plt.plot(meansizes[:1000])
        plt.plot([range_start, range_start], [35, 45], 'r--')
        plt.plot([range_end , range_end], [35, 45], 'r--')
        plt.ylabel('mean size (mm)')
        plt.grid(True)
    
    
        for i, index in enumerate(range(range_start, range_end)):
            im = imread(fpath + fns[index])
            plt.subplot(4,3,i + 4)
            plt.imshow(im, interpolation = 'none', cmap = 'gray')
            plt.xlim([0, 1100])
            plt.ylim([400, 0])
        plt.savefig('plots\\WaveletMethod\\Run354_%d-%d' % (range_start, range_end), bbox_inches = 'tight', dpi = 600)
        plt.close()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        