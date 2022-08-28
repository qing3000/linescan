# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 16:17:00 2022

@author: zhangq
"""
import numpy as np
import matplotlib.pyplot as plt

from imageio import imread
from test_zetica import enhance_image_quality

if __name__ == '__main__':
    fpath = '..\\mud\\Output\\Cribs\\Run_354\\'
    fns, meansizes = np.genfromtxt('meansizes.csv', dtype = str, delimiter = ',', unpack = True, skip_header = 1)
    plt.figure(figsize = (16, 12))
    plt.subplot(4, 1, 1)
    plt.plot(meansizes)
    plt.ylabel('mean size (mm)')
    plt.grid(True)
    while True:
        xy = plt.ginput(1, show_clicks = True)
        print(xy)
        if len(xy) > 0:
            range_start = int(xy[0][0])
            range_end = range_start + 9
            for i, index in enumerate(range(range_start, range_end)):
                im = imread(fpath + fns[index])
                im = enhance_image_quality(im)
                plt.subplot(4,3,i + 4)
                plt.gca().clear()
                plt.imshow(im, interpolation = 'none', cmap = 'gray')
                plt.title('Image#=%d' % index)
                plt.ylim([400, 0])
                plt.draw()
        else:
            break
        # plt.savefig('plots\\WaveletMethod\\Run354_%d-%d' % (range_start, range_end), bbox_inches = 'tight', dpi = 600)
        # plt.close()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        