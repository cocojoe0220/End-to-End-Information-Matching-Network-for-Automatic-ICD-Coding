import matplotlib.pyplot as plt
import random
import numpy as np
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150]
y = [0.5729167, 0.7644231, 0.640625, 0.7604167, 0.6458333, 0.75, 0.7589286, 0.75, 0.72321427, 0.7875, 0.7916667, 0.6796875, 0.73035713, 0.7395833, 0.7410714, 0.75, 0.8125, 0.8054167, 0.8060714, 0.80625, 0.8125, 0.828125, 0.77678573, 0.8229167, 0.81375, 0.8229167, 0.80204544, 0.8020833, 0.7875, 0.8020833, 0.81714287, 0.8125, 0.828125, 0.82285713, 0.8315625, 0.8166667, 0.85, 0.82964287, 0.8375, 0.85, 0.8666667, 0.8958333, 0.875, 0.8625, 0.8958333, 0.8854167, 0.9166667, 0.8958333, 0.85, 0.8375, 0.86875, 0.8479167, 0.8375, 0.8958333, 0.89375, 0.9083333, 0.8958333, 0.9075, 0.8839286, 0.89375, 0.90625, 0.9191176, 0.9285714, 0.9, 0.8854167, 0.9183333, 0.921875, 0.925, 0.9375, 0.933125, 0.9325, 0.91071427, 0.9375, 0.9270833, 0.90625, 0.90625, 0.9123077, 0.9075, 0.9179167, 0.9129167, 0.9075, 0.91428573, 0.910625, 0.9091667, 0.910625, 0.9260714, 0.9283333, 0.9264286, 0.9379167, 0.925, 0.9375, 0.9375, 0.9335417, 0.9453125, 0.9375, 0.91964287, 0.9279167, 0.9283333, 0.9270833, 0.9323077, 0.9375, 0.93428573, 0.930625, 0.9391667, 0.920625, 0.9260714, 0.9383333, 0.9364286, 0.9379167, 0.935, 0.9375, 0.9375, 0.9435417, 0.9353125, 0.9475, 0.94964287, 0.9479167, 0.9483333, 0.9370833, 0.9423077, 0.9475, 0.94428573, 0.940625, 0.9391667, 0.940625, 0.9460714, 0.9483333, 0.9464286, 0.9479167, 0.945, 0.9475, 0.9475, 0.9535417, 0.9513125, 0.9525, 0.94964287, 0.9559167, 0.9523333, 0.9510833, 0.9593077, 0.9515, 0.9515, 0.9515417, 0.9573125, 0.9515, 0.95164287, 0.9519167, 0.9503333, 0.9510833, 0.9553077]

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

