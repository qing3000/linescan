# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 15:57:04 2022

@author: zhangq
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as li
from imageio import imread, imwrite
import pywt
from skimage.restoration import denoise_wavelet, estimate_sigma
import cv2 as cv

import sys
sys.path.append('..\pyDGS')
from dgs import dgs, standardize

def quadratic_surfacefit_mask(zdata, mask):
    M, N = np.shape(zdata)
    xvector = np.arange(0, N) / N #normalisation to avoid large values in high order terms
    yvector = np.arange(0, M) / M #normalisation to avoid large values in high order terms
    xdata, ydata = np.meshgrid(xvector, yvector)
    xdata_masked = xdata[mask]
    ydata_masked = ydata[mask]
    zdata_masked = zdata[mask]
    A = np.zeros((6, 6))
    A[0, 0] = np.sum(np.power(xdata_masked, 4))
    A[0, 1] = A[1, 0] = np.sum(np.power(xdata_masked, 2) * np.power(ydata_masked, 2))
    A[0, 2] = A[2, 0] = np.sum(np.power(xdata_masked, 3) * ydata_masked)
    A[0, 3] = A[3, 0] = np.sum(np.power(xdata_masked, 3))
    A[0, 4] = A[4, 0] = np.sum(np.power(xdata_masked, 2) * ydata_masked)
    A[0, 5] = A[5, 0] = np.sum(np.power(xdata_masked, 2))
    A[1, 1] = np.sum(np.power(ydata_masked, 4))
    A[1, 2] = A[2, 1] = np.sum(xdata_masked * np.power(ydata_masked, 3))
    A[1, 3] = A[3, 1] = np.sum(xdata_masked * np.power(ydata_masked, 2))
    A[1, 4] = A[4, 1] = np.sum(np.power(ydata_masked, 3))
    A[1, 5] = A[5, 1] = np.sum(np.power(ydata_masked, 2))
    A[2, 2] = np.sum(np.power(xdata_masked, 2) * np.power(ydata_masked, 2))
    A[2, 3] = A[3, 2] = np.sum(np.power(xdata_masked, 2) * ydata_masked)
    A[2, 4] = A[4, 2] = np.sum(xdata_masked * np.power(ydata_masked, 2))
    A[2, 5] = A[5, 2] = np.sum(xdata_masked * ydata_masked)
    A[3, 3] = np.sum(np.power(xdata_masked, 2))
    A[3, 4] = A[4, 3] = np.sum(xdata_masked * ydata_masked)
    A[3, 5] = A[5, 3] = np.sum(xdata_masked)
    A[4, 4] = np.sum(np.power(ydata_masked, 2))
    A[4, 5] = A[5, 4] = np.sum(ydata_masked)
    A[5, 5] = len(xdata_masked)
    
    B = np.zeros(6)
    B[0] = np.sum(np.power(xdata_masked, 2) * zdata_masked)
    B[1] = np.sum(np.power(ydata_masked, 2) * zdata_masked)
    B[2] = np.sum(xdata_masked * ydata_masked * zdata_masked)
    B[3] = np.sum(xdata_masked * zdata_masked)
    B[4] = np.sum(ydata_masked * zdata_masked)
    B[5] = np.sum(zdata_masked)
    AI = li.pinvh(A)
    X = np.dot(AI, B)
    a, b, c, d, e, f = X
    zmodel = a * np.power(xdata, 2) + b * np.power(ydata, 2) + c * xdata * ydata + d * xdata + e * ydata + f
    msqe = np.std(zdata - zmodel)
    return zmodel, msqe, X

def rescale(data, rng):
    """
    rescales an input dat between mn and mx
    """
    datamin = min(data.flatten())
    datamax = max(data.flatten())
    return (rng[1] - rng[0]) * (data - datamin) / (datamax - datamin) + rng[0]

'''the maximum scale (grain size) considered by the wavelet is the horizontal width dimension divided by this number
so if your image is 1000 pixels wide and maxscale=4, only grains up to 1000/4 = 250 pixels are considered'''
def calculate_PSD(im, maxscale):
    '''Get the shape'''
    M, N = np.shape(im)
    
    '''Remove background'''
    # mask = np.array([[True] * N] * M)
    # bkg, mse, coefs = quadratic_surfacefit_mask(im, mask)
    # im -= bkg
    
    '''Denoise the image'''
    sigma_est = estimate_sigma(im, channel_axis = None, average_sigmas = True)
    denoisedImg = denoise_wavelet(im, channel_axis = None, rescale_sigma = True, method = 'VisuShrink', mode = 'soft', sigma=sigma_est * 2)

    #Why do we need to do this???
    denoisedImg = rescale(denoisedImg, [0, 255])
    
    '''Choose 100 rows from the image'''
    rows = np.linspace(1, M - 1, 100).astype('int')
    
    '''Calculate the wavelet coefficients'''
    cfs, frequencies = pywt.cwt(denoisedImg[rows], np.arange(1, N / maxscale, 2),  'morl' , .5)
    
    '''Average the power spectrum (square of the coefficients) over pixels'''
    power = np.mean(cfs**2, axis = 2)
    
    '''Calcualte the wavelength'''
    wavelengths = 1.0 / frequencies
    
    '''Calculate the power spectrum density'''
    PSD = power / np.array([wavelengths**2] * len(rows)).T
    
    '''Average over all rows'''
    averagePSD = np.mean(PSD, axis = 1)
    
    '''Ignore the low frequency part (why?)'''
    ind = np.where(wavelengths > 2 * np.pi)[0]
    wavelengths = wavelengths[ind]
    averagePSD = averagePSD[ind]
    
    '''Normalise the PSD'''
    averagePSD = averagePSD / np.sum(averagePSD)
    return wavelengths, averagePSD

#====================================
if __name__ == '__main__':
    
    fn = 'zetica_data\\raw\\crib_0000000_0000000_425_866.png'
    maxscale = 6
        
    '''Read in the image'''
    im_raw = imread(fn)
    im0 = im_raw[20:-20, 20:637]
    im1 = im_raw[20:-20, 637:-20]
    M, N = np.shape(im0)
    imwrite('temp.png', im0)
    
    '''Image processing to derive the particle geometries'''
    # im_display = np.dstack([im0] * 3)
    # im0 = im0.astype('double')
    # mask = np.array([[True] * N] * M)
    # bkg, mse, temp = quadratic_surfacefit_mask(im0, mask)
    
    # im1 = im0 - bkg
    # im2 = cv.GaussianBlur(im1, (3, 3), 0)
    # im3 = rescale(im2, [0, 255]).astype('uint8')
    
    # edgeImg = (im3 > 70).astype('uint8') * 255
    # contours, hierarchy = cv.findContours(edgeImg, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    # validContours = list(filter(lambda x : cv.contourArea(x) > 20, contours))
    
    # blocks = np.zeros((M, N , 3)).astype('uint8')
    # cv.drawContours(blocks, validContours, -1, (255, 255, 255), -1)
    # #cv.drawContours(im_display, validContours, -1, (105, 0, 0), 1)
    
    # #plt.imsave('test.png', blocks)
    # #a = Annotate(edgeImg)
    # plt.subplot(2,2,1)
    # plt.imshow(im_display, interpolation = 'none', cmap = 'gray')
    # plt.title('Raw image')
    # plt.subplot(2,2,2)
    # plt.imshow(im3, interpolation = 'none', cmap = 'jet', vmin = 0, vmax = 100)
    # plt.title('Background corrected and Gaussian smoothed')
    # plt.colorbar()
    # plt.subplot(2,2,3)
    # plt.imshow(edgeImg, interpolation = 'none', cmap = 'gray')
    # plt.title('Thresholded image')
    # plt.subplot(2,2,4)
    # plt.imshow(blocks, interpolation = 'none', cmap = 'gray')
    # plt.title('Clean up')
    # raise SystemExit
    
    im = im0.astype('double')
    
    wavelengths, psd = calculate_PSD(im, maxscale)
    
    resolution = 1435.0 / N
    
    data_out = dgs('temp.png', resolution = 1, maxscale = maxscale, verbose = 0, x = 0)
    hx = data_out['grain size bins']
    hy = data_out['grain size frequencies']
    
    plt.plot(wavelengths, psd)
    plt.plot(hx, hy)
    plt.xlabel('Wavelength (pixels)')
    plt.ylabel('PSD')
    plt.grid(True)


       
       