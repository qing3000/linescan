# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 11:13:46 2022

@author: zhangq
"""

# import numpy as np
# import matplotlib.pyplot as plt

# p = np.arange(0.1, 0.9, 0.01)
# y = (1 - p)**8

# plt.plot(p, y)
# plt.grid(True)


import numpy as np
import matplotlib.pyplot as plt
import pywt
from skimage.restoration import denoise_wavelet, estimate_sigma
import test_zetica as tt

fn = 'zetica_data\\raw\\crib_0000000_0000000_425_866.png'
im0 = plt.imread(fn)
im = im0[20:-20, 20:637]
M, N = np.shape(im)

mask = np.array([[True] * N] * M)
bkg, mse, temp = tt.quadratic_surfacefit_mask(im, mask)
im -= bkg

sigma_est = estimate_sigma(im, channel_axis = None, average_sigmas = True)
denoisedImg = denoise_wavelet(im, channel_axis = None, rescale_sigma = True, method = 'VisuShrink', mode = 'soft', sigma=sigma_est * 2)


row = 200

y0 = im[row].astype('float')
y = denoisedImg[row].astype('float')
t = np.arange(len(y))
maxscale = 80
wavelet_name = 'morl'


wavelet = pywt.ContinuousWavelet(wavelet_name)
[psi, x] = wavelet.wavefun(8)
scales = np.arange(1, maxscale)

'''Contineous wavelet transform. The frequency is calculated as wavelet_centre_frequency / scales'''
coef, frequencies = pywt.cwt(y, scales, wavelet_name)
wavelengths = 1 / frequencies
averagePSD = np.mean(coef**2, 1) / wavelengths**2
averagePSD = averagePSD / np.sum(averagePSD)

waves, fullPSD = tt.calculate_PSD(im, maxscale)
plt.subplot(2,2,1)
plt.imshow(im, interpolation = 'none', aspect = 'auto', cmap = 'gray')
plt.plot([0, N - 1], [row, row], ':')
plt.subplot(2,2,2)
plt.plot(t, y0, label = 'Raw signal')
plt.plot(t, y, label = 'Wavelet denoised signal')
plt.plot(x * 20 + t[-1] / 2, psi, ':', label = 'Morlet wavelet (20 x scaled up)')
plt.plot(x * 40 + t[-1] / 4, psi, ':', label = 'Morlet wavelet (40 x scaled up)')
plt.grid(True)
plt.legend(loc = 0)
plt.title('Sinusoidal wave and Gauss wavelet')
plt.xlim([t[0], t[-1]])
plt.subplot(2,2,3)
plt.imshow(coef, interpolation = 'none', aspect = 'auto', cmap = 'PuOr', extent = [t[0], t[-1], scales[-1], scales[0]])
plt.ylabel('Scale')
plt.title('Contineous wavelet transform')
plt.subplot(2,2,4)
plt.plot(averagePSD, scales)
plt.plot(fullPSD, scales)
plt.grid(True)
plt.title('Power spectrum density')
plt.gca().invert_yaxis()
plt.xlabel('Frequency (Hz)')
