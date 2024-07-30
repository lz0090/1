
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax1 = plt.axes(projection='3d')
gama_1 = 1.5
gama_2 = 1
delay = ([[43.9625, 44.2855, 44.3385, 44.6285, 45.5853],
          [44.1758, 44.3384, 44.4191, 45.5251, 46.0555],
          [44.4313, 45.2480, 45.8331, 45.9069, 45.9931],
          [44.3709, 45.0028, 45.4416, 45.6737, 45.8331],
          [43.9907, 44.3672, 44.6621, 44.9388, 45.3461]
])
num_all = ([[36, 20, 13, 11, 8],
            [30, 18, 14, 12, 4],
            [27, 12, 8, 5, 6],
            [27, 18, 11, 10, 8],
            [29, 19, 13, 16, 13]
])
Q = np.zeros((5, 5))
for i in range(5):
    for j in range(5):
        Q[i][j] = gama_1 * delay[i][j] / 58 + gama_2 * num_all[i][j] / 50

print(Q)

x = ([[0.1, 0.1, 0.1, 0.1, 0.1],
            [0.3, 0.3, 0.3, 0.3, 0.3],
            [0.5, 0.5, 0.5, 0.5, 0.5],
            [0.7, 0.7, 0.7, 0.7, 0.7],
            [0.9, 0.9, 0.9, 0.9, 0.9]
])

y = ([[0.05, 0.15, 0.25, 0.35, 0.45],
            [0.05, 0.15, 0.25, 0.35, 0.45],
            [0.05, 0.15, 0.25, 0.35, 0.45],
            [0.05, 0.15, 0.25, 0.35, 0.45],
            [0.05, 0.15, 0.25, 0.35, 0.45]
])

ax1.plot_surface(x, y, Q,cmap='rainbow')
#ax3.contour(X,Y,Z, zdim='z',offset=-2，cmap='rainbow)   #等高线图，要设置offset，为Z的最小值
plt.show()