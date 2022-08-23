# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 22:18:57 2022

@author: zhangq
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fft2, fftshift, fftfreq, ifftshift, ifft2

N = 500
y = np.sin(2 * 10 * np.pi / N * np.arange(N))
fy = fft(y, 1024)[:int(N / 2)]
x = N / np.arange(len(fy)) / 2

plt.subplot(3,1,1)
plt.plot(y)
plt.grid(True)
plt.title('Sinosuidal wave (10 waves)')
plt.subplot(3,1,2)
plt.plot(np.abs(fy))
plt.xlim([0, 30])
plt.grid(True)
plt.xlabel('Number of waves')
plt.title('Fourier Transform')
plt.subplot(3,1,3)
plt.plot(x, np.abs(fy), '.-')
plt.grid(True)
plt.xlabel('Wave size (pixels)')

plt.xlim([0, 30])
raise SystemExit

M = 101
N = 101
fim = np.zeros((M, N))
cc = [int(M / 2 + 0.5) - 1, int(N / 2 + 0.5) - 1]
offset = 5
offset1 = 5
fim[cc[0] - offset, cc[1]] = 1
fim[cc[0] + offset, cc[1]] = 1
fim[cc[0], cc[1] - offset] = 1
fim[cc[0], cc[1] + offset] = 1

fim[cc[0] - offset1, cc[1] - offset1] = 1
fim[cc[0] - offset1, cc[1] + offset1] = 1
fim[cc[0] + offset1, cc[1] - offset1] = 1
fim[cc[0] + offset1, cc[1] + offset1] = 1
im = ifft2(ifftshift(fim))
plt.figure()
plt.subplot(1,2,1)
plt.imshow(fim, cmap = 'gray_r', extent = [-cc[0], cc[0], cc[1], -cc[1]])
plt.title('Frequency domain')
plt.grid(True)
plt.subplot(1,2,2)
plt.imshow(np.real(im))
plt.title('Inverse 2D FFT (real part)')



