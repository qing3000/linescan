# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 11:13:46 2022

@author: zhangq
"""

import numpy as np
from numpy.fft import fft, fftshift, fftfreq
import matplotlib.pyplot as plt

f = 0.005 #Hz
N = 1000
t = np.arange(N)
y = np.sin(2 * np.pi * t * f)

yf = fft(y)
yf = fftshift(yf)
freq = fftfreq(N, 1)
freq = fftshift(freq)

plt.subplot(3,1,1)
plt.plot(t, y)
plt.grid(True)
plt.title(r'$sin(2\pi f t), f=%gHz$' % f)
plt.subplot(3,1,2)
plt.plot(freq, np.abs(yf))
plt.grid(True)
plt.title('Fourier transform magnitude')
plt.subplot(3,1,3)
plt.plot(freq, np.unwrap(np.angle(yf)) / np.pi / 2)
plt.grid(True)
plt.title('Fourier transform phase')
plt.xlabel('Frequency (Hz)')
