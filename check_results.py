# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 11:26:03 2022

@author: zhangq
"""

import numpy as np
import matplotlib.pyplot as plt

L = 100
s1, s2 = np.loadtxt('sizemeans.csv', delimiter = ',', unpack = True)
ss1 = np.convolve(s1, np.ones(L) / L, mode = 'valid')
ss2 = np.convolve(s2, np.ones(L) / L, mode = 'valid')
plt.plot(ss1)
plt.plot(ss2)
plt.grid(True)