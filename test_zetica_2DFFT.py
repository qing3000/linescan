# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 15:57:04 2022

@author: zhangq
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as li
from scipy.interpolate import griddata
from numpy.fft import rfft, fft2, rfft2, fftshift, rfftfreq
from imageio import imread
import glob
from scipy.interpolate import interp1d
import scipy.ndimage as ni

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

def turn_square(im):
    M, N = np.shape(im)
    m_upper = int((N - M) / 2 + 0.5)
    m_lower = N - M - m_upper
    upper_part = im[:m_upper,:]
    lower_part = im[-m_lower:, :]
    im_square = np.concatenate((upper_part[::-1, :], im, lower_part[::-1, :]))
    return im_square

def fixedHist(x, binWidth):
    binEdges = np.arange(np.min(x) - binWidth / 2.0, np.max(x) + binWidth / 2.0, binWidth)
    if binEdges[-1] < np.max(x): 
        binEdges = np.append(binEdges, np.max(x) + binWidth / 2.0)
    hy, binEdges = np.histogram(x, binEdges, density = True)
    hx = (binEdges[:-1] + binEdges[1:]) / 2
    return hx, hy

#====================================
if __name__ == '__main__':
    fn = 'C:\\Personal\\Python\\Zetica\\linescan\\zetica_data\manual_secondary_lengths.csv'
    data = np.genfromtxt(fn, delimiter = ',', skip_header = 1)

    # plt.bar(hx, hy, width = binsize - 0.2)
    # plt.grid(True)
    # raise SystemExit
    
    
    fpath = 'C:\\Personal\\Python\\Zetica\\mud\\Data\\RioTinto\\all_cribs\\Run_122-20211004@162148\\'
    fpath = 'C:\\Personal\\Python\\Zetica\\mud\\Data\\RioTinto\\mudspot_cribs\\Run_122-20211004@162148\\'
    fpath = 'C:\\Personal\\Python\\Zetica\\linescan\\zetica_data\\cribimages\\'
    fns = glob.glob(fpath + '*.png')
    #fns = glob.glob('zetica_data\\raw\\*.png')
    for i, fn in enumerate(fns):
            
        '''Read in the image'''
        im_raw = imread(fn)
        # im_left = im_raw[20:-20, 20:617]
        # im_right = im_raw[20:-20, 637:-20]
        im_full = im_raw[25:-30, :-20]
        
        im = im_full
        im_enh = enhance_image_quality(im)
        # hx, hy = fixedHist(im_enh.flat, 2)
        # plt.plot(hx, hy)
        # plt.grid(True)
        # raise SystemExit
        #im_sqr = turn_square(im)
        #M, N = np.shape(im_sqr)
        #im_sqr_enh = enhance_image_quality(im_sqr)
    
        # plt.subplot(1,2,1)
        # plt.imshow(im_disp, cmap = 'gray', interpolation = 'none')
        # plt.subplot(1,2,2)
        # plt.imshow(im_sqr_disp, cmap = 'gray', interpolation = 'none')
        # raise SystemExit
        im1 = im_enh.astype('double')
        # mask = np.array([[True] * N] * M)
        # bkg, mse, temp = quadratic_surfacefit_mask(im0, mask)
        # im1 = im0 - bkg
        im2 = im1 - np.mean(im1)
        
        # onerow = im2[300, :]
        # x = rfftfreq(N)
        # y = rfft(onerow)
        # plt.subplot(2,1,1)
        # plt.plot(onerow)
        # plt.grid(True)
        # plt.subplot(2,1,2)
        # plt.plot(x, np.abs(y))
        # plt.grid(True)
        # raise SystemExit
        M, N = np.shape(im2)
        sfim = fftshift(fft2(im2))
        sfim_mag = np.abs(sfim)
        # x = np.linspace(0, M, M)
        # f = interp1d(x, sfim_mag.T, axis = 1)
        # xi = np.linspace(0, M, N)
        # sfimi_mag = f(xi)
        sfimi_mag = np.abs(fftshift(fft2(im2, (N, N))))
        # print((M, N))
        # print(sfim_mag[int(M / 2), int(N / 2)])
        # #print(sfimi_mag[int(N / 2), int(N / 2)])
        # print(sfimi1_mag[int(N / 2), int(N / 2)])
        # continue

        
        rows, cols = np.nonzero(sfimi_mag >= 0)
        pnts = np.array((cols, rows)).T
        values = sfimi_mag.flat
        M, N = np.shape(sfimi_mag)
        cc = int(N / 2 )
        
        theta = 0
        r = np.arange(0, N / 2)
        thetas = np.linspace(0, np.pi, 180)
        T = np.array([thetas] * len(r)).T
        R = np.array([r] * len(thetas))
        gx = R * np.cos(T) + cc
        gy = R * np.sin(T) + cc
        #raise SystemExit
        print('grid start')
        rays = griddata(pnts, values, (gx.flat, gy.flat), method = 'linear')
        print('grid end')
        rays = np.reshape(rays, (len(thetas), -1))
        averageRay = np.mean(rays, 0)
        x = N / r[1:] / 2
        y = averageRay[1:]
        #y[x > 200] = 0
        xi = np.arange(x[-1], x[0])
        f = interp1d(x, y, kind = 'linear')
        yi = f(xi)
        plt.figure(figsize = (12, 8))
        plt.subplot(2,2,3)
        plt.plot(averageRay / np.sum(averageRay))
        plt.grid(True)
        plt.xlabel('Number of waves')
        plt.title('Averaged radial line from 2D-FFT')
        plt.xlim([0, 50])
        plt.subplot(2,2,4)
        plt.plot(x / 4, y / np.max(yi), '.-', label = '2D-FFT')
        
        x = data[:, i]
        x = x[np.logical_not(np.isnan(x))]
        binsize = 5
        binEdges = np.arange(0, 100, 5)
    
        hy, binEdges = np.histogram(x, binEdges, density = True)
        hx = (binEdges[:-1] + binEdges[1:]) / 2
        plt.plot(hx, hy / np.max(hy), label = 'Manual')
        plt.grid(True)
        plt.legend(loc = 0)
        plt.xlabel('Particle size(pixels)')
        plt.title('PSD')
        
        plt.subplot(2,2,1)
        plt.imshow(im_enh, interpolation = 'none', cmap = 'gray')
        plt.title(fn[fn.rfind('\\') + 1:-4])
        plt.subplot(2,2,2)
        plt.imshow(sfimi_mag, cmap = 'jet', extent = [-N / 2, N / 2, N / 2, -N / 2], vmax = 1e6)
        plt.xlim([-50, 50])
        plt.ylim([50, -50])
        plt.title('2D FFT')
        shortfn = fn[fn.rfind('\\') + 1:-4]
        plt.savefig('%s.png' % shortfn, dpi = 400, bbox_inches = 'tight')
        plt.close()
    raise SystemExit
    x = rfftfreq(N)
    y = rfftfreq(M)
    z = rfft(im2[:, 300])
    fim_mag = np.abs(sfim)
    
    plt.subplot(2,2,1)
    plt.imshow(im1, interpolation = 'none', cmap = 'gray')
    plt.subplot(2,2,2)
    plt.plot(im2[:, 300], range(M))
    plt.grid(True)

    plt.subplot(2,2,3)
    plt.plot(x, fim_mag[int(M / 2 + 0.5), :])
    plt.grid(True)
    
    plt.subplot(2,2,4)

       
       