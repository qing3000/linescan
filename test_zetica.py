# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 15:57:04 2022

@author: zhangq
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as li
from imageio import imread
import pywt
from skimage.restoration import denoise_wavelet, estimate_sigma
import glob
import scipy.ndimage as ni
from datetime import datetime

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
def calculate_PSD_buscombe(im, maxscale):
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
    #Why 0.5 sampling period???
    cfs, frequencies = pywt.cwt(denoisedImg[rows], np.arange(1, N / maxscale, 2),  'morl' , .5)
    
    '''Average the power spectrum (square of the coefficients) over pixels'''
    powerSpectrum = np.mean(cfs**2, axis = 2)

    '''Average over all rows'''
    averagePowerSpectrum = np.mean(powerSpectrum, axis = 1)
    
    '''Calculate the wavelength'''
    wavelengths = 1.0 / frequencies
    
    '''Calculate the PSD'''
    averagePSD = averagePowerSpectrum / wavelengths**2
    
    '''Ignore the low frequency part (why?)'''
    ind = np.where(wavelengths > 2 * np.pi)[0]
    wavelengths = wavelengths[ind]
    averagePSD = averagePSD[ind]
    
    '''Normalise the PSD'''
    averagePSD = averagePSD / np.sum(averagePSD)
    return wavelengths, averagePSD

'''the maximum scale (grain size) considered by the wavelet is the horizontal width dimension divided by this number
so if your image is 1000 pixels wide and maxscale=4, only grains up to 1000/4 = 250 pixels are considered'''
def calculate_PSD(im, maxscale):
    '''Remove background'''
    # mask = np.array([[True] * N] * M)
    # bkg, mse, coefs = quadratic_surfacefit_mask(im, mask)
    # im -= bkg
    
    '''Denoise the image'''
    sigma_est = estimate_sigma(im, channel_axis = None, average_sigmas = True)
    denoisedImg = denoise_wavelet(im, channel_axis = None, rescale_sigma = True, method = 'VisuShrink', mode = 'soft', sigma=sigma_est * 2)
    
    #Why do we need to do this???
    denoisedImg = rescale(denoisedImg, [0, 255])
    
    '''Calculate the wavelet coefficients'''
    cfs, frequencies = pywt.cwt(denoisedImg, np.arange(1, maxscale),  'morl')
    
    '''Calculate the wavelength'''
    wavelengths = 1.0 / frequencies
    
    '''Calculate the PSD'''
    M, N = np.shape(im)
    psd = (cfs / np.array([[wavelengths] * M] * N).T) **2
    
    '''Average the psd over columns'''
    averagePSD1 = np.mean(psd, axis = 2)
    
    '''Average the psd over rows'''
    #'''Choose 100 rows from the image'''
    #rows = np.linspace(0, M - 1, 100).astype('int')
    averagePSD = np.mean(averagePSD1, axis = 1)

    '''Plot the wavelet transform'''
    # rows = np.linspace(50, M - 50, 3).astype('int')
    # wavelet = pywt.ContinuousWavelet('morl')
    # [psi, x] = wavelet.wavefun(8)
    # plt.subplot(3,1,1)
    # plt.imshow(im, interpolation = 'none', cmap = 'gray')
    # plt.plot([0, N -1], [rows, rows], ':r')
    # plt.title('Rows at red dash linse')
    # for i, row in enumerate(rows):
    #     plt.subplot(3,3,4 + i)
    #     plt.plot(denoisedImg[row])
    #     plt.plot(x * 20 + N / 2, psi * 100 + np.mean(im[row]), '--', label = 'Morlet wavelet (10 x scaled up)')
    #     plt.xlim([0, N])
    #     plt.grid(True)
    #     plt.legend(loc = 0)
    #     plt.title('Intensity profile of Row %d' % row)
    #     plt.subplot(3,3,7 + i)
    #     plt.imshow(psd[:, row, :], interpolation = 'none', aspect = 'auto', vmax = 30, cmap = 'jet')
    #     plt.title('Wavelet PSD of row %d' % row)
    #     plt.xlabel('Column')
    #     plt.ylabel('Scale')
    
    '''Plot the average PSD'''
    # for row in rows:
    #     plt.plot(wavelengths, averagePSD1[:, row], label = 'Row %d' % row)
    # plt.legend(loc = 0)

    # plt.grid(True)
    # plt.xlabel('Wavelengths (pixels)')
    # plt.ylabel('Magntidue^2')
    # plt.title('Average PSD')
    # raise SystemExit

    '''Normalise the PSD'''
    averagePSD = averagePSD / np.sum(averagePSD)
    return wavelengths, averagePSD

def enhance_image_quality(im0):
    contrastNSigma = 3.5
    smoothWinSize = 10
    im = im0.astype('float')
    M, N = np.shape(im)    
    p1 = np.mean(im, 0)
    p2 = ni.convolve(p1, np.ones(smoothWinSize) / smoothWinSize, mode = 'mirror')    
    intensityMatrix = np.array([p2] * M)
    im2 = im / intensityMatrix - 1
    rowSigmas = np.std(im2, axis = 1)
    averageSigma = np.mean(rowSigmas)
    contrastRatio = 128 / (averageSigma * contrastNSigma)
    im3 = im2 * contrastRatio + 128;
    im3[im3 < 0] = 0
    im3[im3 > 254] = 254
    # plt.subplot(3,1,1)
    # plt.imshow(im, interpolation = 'none', cmap = 'gray')
    # plt.grid(True)
    # plt.title('Raw crib image')
    # plt.subplot(3,1,2)
    # plt.plot(p2)
    # plt.title('Intensity profile')
    # plt.xlim([0, N - 1])
    # plt.grid(True)
    # plt.subplot(3,1,3)
    # plt.imshow(im3, interpolation = 'none', cmap = 'gray')
    # plt.grid(True)
    # plt.title('Contrast enhanced')
    # raise SystemExit
    return im3

#====================================
if __name__ == '__main__':
    fns = glob.glob('zetica_data\\cribimages\\*.png')
    wave_means = []
    manual_means = []
    maxscale = 80
            
    for i, fn in enumerate(fns):

        '''Read in the image'''
        im_raw = imread(fn)
        im = im_raw[25:-30, :-20]
        
        im = enhance_image_quality(im)
        
        wavelengths, psd = calculate_PSD(im.astype('double'), maxscale)
        meansize = np.sum(psd * wavelengths)
        
        
        csvfn = 'zetica_data\\cribimages\\manual_secondary_lengths.csv'
        data = np.genfromtxt(csvfn, delimiter = ',', skip_header = 1)
        x = data[:, i]
        x = x[np.logical_not(np.isnan(x))]
        binsize = 5
        binEdges = np.arange(0, 100, 5)
    
        hy, binEdges = np.histogram(x, binEdges, density = True)
        hx = (binEdges[:-1] + binEdges[1:]) / 2
        wave_means.append(meansize)
        manual_means.append(np.mean(x))
        plt.subplot(2,1,1)
        plt.imshow(im, interpolation = 'none', cmap = 'gray')
        plt.subplot(2,1,2)
        plt.plot(wavelengths, psd, label = 'wavelet method mean=%.0f pixels' % meansize)
        plt.plot(hx, hy, label = 'manual method mean=%.0f pixels' % np.mean(x))
        plt.xlabel('Wavelength (pixels)')
        plt.ylabel('PSD')
        plt.grid(True)
        plt.legend(loc = 0)
        plt.title('Particle size distributions')
        shortfn = fn[fn.rfind('\\') + 1:-4]
        plt.savefig('plots\\WaveletMethod\\%s.png' % shortfn, bbox_inches = 'tight')
        plt.close()

    p = np.polyfit(manual_means[:-1], wave_means[:-1], 1)
    x = np.arange(15, 35)
    plt.plot(manual_means, wave_means, '.')#
    plt.plot(x, np.polyval(p, x), '--')
    plt.grid(True)
    plt.xlabel('Average particle size (secondary axis) by manual method')
    plt.ylabel('Average particle size by wavelet method')
    plt.title('Correlation')
       