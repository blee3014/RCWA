import scipy.constants as cte
import numpy as np
from conv import convmat2D
import rcwa

if __name__ == '__main__':
    mat_ref = (1.0, 2.0)
    mat_trn = (1.0, 9.0)
    angles = (30, 20)
    wavelength = 2.0
    unit = 1e-2
    pol = (0, 1)
    InitialVal = rcwa.Initial(mat_ref,
                              mat_trn,
                              angles,
                              wavelength,
                              unit,
                              pol
                              )



# # Step 0: Define the Problem. Refer to https://www.youtube.com/watch?v=n4Jy4TWHBZM
# # for the simulation problem.
# # define units
# um = 1e-6
# cm = 1e-2
# nm = 1e-9
#
# # Material properties
# # Reflection region
# u_ref = 1.0
# eps_ref = 2.0
#
# # device region
# u_r = 1.0
# eps_r = 6.0
#
# # transmission region
# u_trn = 1.0
# eps_trn = 9.0
#
# # Unit cell size
# Lx = 1.75*cm # cm
# Ly = 1.50*cm # cm
#
# # layer thickness
# d1 = 0.50*cm # cm
# d2 = 0.30*cm # cm
#
# # triangle size in layer 1
# w = 0.8*Ly
#
# # wavelength. Field will be linearly polarized along y axis
# lambda_0 = 2.0*cm       # cm
# theta_ang_inc = 45      # degrees
# psi_ang_inc = 0.0       # degrees
#
# theta_ang_inc = np.deg2rad(theta_ang_inc)
# psi_ang_inc = np.deg2rad(psi_ang_inc)
#
# polarization = 'TE'     # or 'TM'
#
# # define constants
# c0 = cte.speed_of_light
# u0 = cte.mu_0
# eps0 = cte.epsilon_0
# eta0 = np.sqrt(u0/eps0)
#
# # Step 2: Build device on grid
# num_x = 256
# num_y = num_x
#
# dx = Lx / num_x
# dy = Ly / num_y
# xa = np.arange(0, num_x) * dx
# ya = np.arange(0, num_y) * dy
# xa = xa - np.average(xa)
# ya = ya - np.average(ya)
# X, Y = np.meshgrid(xa, ya)
#
# # Grid indices of the triangle
# h = w * np.sqrt(3.0) / 2
# ny = int(np.round(h / dy))
# ny1 = int(np.floor((num_y - ny) / 2.0))
# ny2 = ny1 + ny - 1
#
# EPS_DEV = np.ones((num_x, num_y))*eps_r
# U_DEV = np.ones((num_x, num_y))*u_r
#
# for ny in range(ny1, ny2 + 1):
#     f = 1 - float((ny - ny1 + 1)) / (ny2 - ny1 + 1)
#     nx = int(np.round(f * w / dx))
#     nx1 = int(np.floor((num_x - nx) / 2.0))
#     nx2 = nx1 + nx - 1
#     EPS_DEV[nx1:nx2, ny - 1] = eps_ref
#     U_DEV[nx1:nx2, ny - 1] = u_ref
#
# # Step 3: Compute Convolution Matrices
# num_harmonics_x = int(7*(Lx/lambda_0))
# num_harmonics_y = int(7*(Ly/lambda_0))
#
# EPS_DEV_C = convmat2D(EPS_DEV, *(num_harmonics_x, num_harmonics_y))
# U_DEV_C = convmat2D(U_DEV, *(num_harmonics_x, num_harmonics_y))
#
# # # plot the device's eps_r, u_r and convolution matrices
# # plt.subplot(221)
# # plt.pcolormesh(xa, ya, EPS_DEV.T, cmap='RdBu_r')
# # plt.axis('equal')
# # plt.title('UNIT CELL ($\epsilon_r$)')
# # plt.colorbar()
# #
# # plt.subplot(222)
# # plt.pcolormesh(xa, ya, U_DEV.T, cmap='RdBu_r')
# # plt.axis('equal')
# # plt.title('UNIT CELL ($\mu_r$)')
# # plt.colorbar()
# #
# # plt.subplot(223)
# # plt.imshow(np.real(EPS_DEV_C.T), interpolation='none')
# # plt.axis('equal')
# # plt.title('$\epsilon_r$ CONVOLUTION MATRIX')
# # plt.colorbar()
# #
# # plt.subplot(224)
# # plt.imshow(np.real(U_DEV_C.T), interpolation='none')
# # plt.axis('equal')
# # plt.title('$\mu_r$ CONVOLUTION MATRIX')
# # plt.colorbar()
# # plt.show()
#
# # Step 4: Compute Wave Vector Expansion, and normalize with k0
# k0 = 2*np.pi/lambda_0
# k_inc_norm = np.sqrt(u_ref * eps_ref) * np.array([np.sin(theta_ang_inc)*np.cos(psi_ang_inc),
#                                                   np.sin(theta_ang_inc)*np.sin(psi_ang_inc),
#                                                   np.cos(theta_ang_inc)
#                                                   ]
#                                                  )
# kx_norm = k_inc_norm[0] - 2*np.pi*np.arange(-(num_harmonics_x/2), num_harmonics_x/2 + 1)/(k0*Lx)
# ky_norm = k_inc_norm[1] - 2*np.pi*np.arange(-(num_harmonics_y/2), num_harmonics_y/2 + 1)/(k0*Ly)
# KX_MESH, KY_MESH = np.meshgrid(kx_norm, ky_norm)
#
# KX = dia_matrix((kx_norm, [0]), shape=(num_harmonics_x, num_harmonics_x))
# KY = dia_matrix((ky_norm, [0]), shape=(num_harmonics_y, num_harmonics_y))
#
# # Todo: Calculate Longitudinal wave vector components in Region I and Region II
# # kz_ref = np.conj(np.sqrt(k_inc**2 * u_ref * eps_ref - KX_MESH**2 - KY_MESH**2))
# # kz_trn = np.conj(np.sqrt(k_inc**2 * u_trn * eps_trn - KX_MESH**2 - KY_MESH**2))
#
#
# # Step 5: Compute Eigenmodes of Free Space




