# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:58:28 2019

@author: Kartik
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})
#
# X=[27.6,32.5,38.2,45.4,52.5,56.5,60.3,63.6,68,70,74,78.2]  # payload
# Y=[34.3,38.5,43.2,47.8,55.5,59.5,63.8,67.9,71,76.2,80.3,84.8] # header+ payload
# Z=[100,200,300,500,600,700,800,900,1000,1100,1200,1500]


X = [27.6, 32.5, 38.2, 41.7, 45.4, 52.5, 56.5, 60.3, 63.6, 68, 70, 74, 76.3, 77.4, 78.2]  # payload
Y = [34.3, 38.5, 43.2, 45.2, 47.8, 55.5, 59.5, 63.8, 67.9, 71, 76.2, 80.3, 81.5, 83, 84.8]  # header+ payload
Z = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]

plt.plot(Z, Y, 'ro-')
plt.plot(Z, X, 'g*-')

# plt.grid()
plt.ylim(0, 100)
plt.ylabel('Classification accuracy (%)')
plt.xlabel('Input size (bytes) of DT')
plt.legend(['Header + Payload', 'Payload'], loc='lower right')
plt.savefig('Accfinal_11.jpg', dpi=1000)
plt.savefig('packets_based_resluts.pdf')
plt.show()
