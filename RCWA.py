"""
Step 0: Define the problem
Step 1: Initialize Program
Step 2: Build device on grid
Step 3: Compute convolution matrices
Step 4: Compute wave vector expansion
Step 5: Compute eigen-modes of free space
Step 6: Initialize global scattering matrix
Step 7: Main loop through layers
        1. Compute P and Q
        2. Compute eigen-modes
        3. Compute layer scattering matrix
        4. Update global scattering matrix

Step 8: Compute reflection side
Step 9: Compute transmission side scattering matrix
Step 10: Update global scattering matrix
Step 11: Compute reflected and transmitted fields
Step 12: Compute diffraction efficiencies
Step 13: Verify conservation of energy
"""

import scipy.constants as cst
import numpy as np
import matplotlib.pyplot as plt
from fft_coefficients import convmat2D

# Step 0: Define the Problem. Refer to https://www.youtube.com/watch?v=n4Jy4TWHBZM
# for the simulation problem.

# Material properties
# Reflection region
u_ref = 1.0
eps_ref = 2.0

# device region
u_r = 1.0
eps_r = 6.0

# transmission region
u_trn = 1.0
eps_trn = 9.0

# Unit cell size
K_x = 1.75 # cm
K_y = 1.50 # cm

# layer thickness
d1 = 0.5 # cm
d2 = 0.3 # cm

# triangle size in layer 1
w = 0.8*K_y

# wavelength
lambda_0 = 2.0 # cm

# define units
um = 1e-6
cm = 1e-2
nm = 1e-9

# define constants
c0 = cst.speed_of_light
u0 = cst.mu_0
eps0 = cst.epsilon_0
eta0 = np.sqrt(u0/eps0)

# Step 2: Build device on grid
num_x = 512
num_y = num_x

w = 0.9*K_x
dx = K_x / num_x
dy = K_y / num_y
xa = np.arange(0, num_x) * dx
ya = np.arange(0, num_y) * dy
xa = xa - np.average(xa)
ya = ya - np.average(ya)
X, Y = np.meshgrid(xa, ya)

# Grid indices of the triangle
h = w * np.sqrt(3.0) / 2
ny = int(np.round(h / dy))
ny1 = int(np.floor((num_y - ny) / 2.0))
ny2 = ny1 + ny - 1

ER_T = np.ones((num_x, num_y))*eps_ref
for ny in range(ny1, ny2 + 1):
    f = 1 - float((ny - ny1 + 1)) / (ny2 - ny1 + 1)
    nx = int(np.round(f * w / dx))
    nx1 = int(np.floor((num_x - nx) / 2.0))
    nx2 = nx1 + nx - 1
    ER_T[nx1:nx2, ny - 1] = eps_r

plt.subplot(1,1,1)
plt.pcolormesh(X, Y, ER_T.T, cmap='RdBu_r')
plt.axis('equal')
plt.colorbar()
plt.show()
# Step 3: Compute Convolution Matrices




