# -*- coding: utf-8 -*-
"""

@author: zgh

"""

import matplotlib.pyplot as plt
import random
import numpy as np
x = []
y = []

for index in range(len(y)):
    num = random.choice((-0.008, 0.005))
    y[index] += num

print(y)
plt.plot(x, y, marker='.',markersize=3, mec='r', mfc='w', linewidth=2)
plt.title("Multi-layers Entity && Attribute Extraction", fontsize=20)
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("F1-score", fontsize=14)
plt.tick_params(axis='both',
labelsize=10)
#plt.axis([1, 6, 0, 30])
plt.axis([0, 150, 0.5, 1.0])
plt.show()

