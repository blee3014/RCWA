import scipy.constants as cte
from scipy.sparse import dia_matrix
import scipy as sc
import numpy as np
from conv import convmat2D
import rcwa
import matplotlib.pyplot as plt
from copy import deepcopy

if __name__ == '__main__':
    # ------ Step 1: Initialize Program ---------------------------------------------------
    # define constants
    c0 = cte.speed_of_light     # speed of light
    u0 = cte.mu_0               # permeability of free space
    eps0 = cte.epsilon_0        # permittivity of free space
    eta0 = np.sqrt(u0/eps0)     # impedance of free space

    # Initial parameters
    unit = 1e-2                         # units = cm
    u_ref, eps_ref = (1.0, 2.0)         # material property of reflection region
    u_trn, eps_trn = (1.0, 9.0)         # material property of transmission region
    theta, psi = (0, 0)                 # incident angles
    lambda_0 = 2.0                      # wavelength (cm)
    k0 = 2 * np.pi / (lambda_0*unit)    # wavenumber
    pol = (1, 0)                        # polarization = [TE, TM]

    # device region
    u_r = 1.0                   # permeability of device triangle
    eps_r = 6.0                 # permittivity of device triangle

    # Unit cell size
    Lx = 1.75                   # unit cell length in x (cm)
    Ly = 1.50                   # unit cell length in y (cm)

    # layer thickness
    d1 = 0.50                   # first layer thickness (cm)
    d2 = 0.30                   # second layer thickness (cm)

    # Num of harmonics
    num_harmonics_x = int(7 * (Lx / lambda_0))
    num_harmonics_y = int(7 * (Ly / lambda_0))

    # Use odd number of harmonics
    if num_harmonics_x % 2 == 0:
        num_harmonics_x += 1
    if num_harmonics_y % 2 == 0:
        num_harmonics_y += 1

    # num_harmonics_x = 3
    # num_harmonics_y = 3

    MN = num_harmonics_x * num_harmonics_y

    # ----- Step 2: Build device on grid --------------------------------------------------
    # triangle size in layer 1
    w = 0.8*Ly*unit

    num_x = 1024    # x, y resolution: 1000 pts
    num_y = int(np.round(num_x * Ly/Lx))

    dx = Lx * unit / num_x
    dy = Ly * unit/ num_y
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

    # Layer 1
    EPS_DEV_1 = np.ones((num_x, num_y)) * eps_r
    U_DEV_1 = np.ones((num_x, num_y)) * u_r

    # Layer 2
    EPS_DEV_2 = deepcopy(EPS_DEV_1)
    U_DEV_2 = deepcopy(U_DEV_1)

    for ny in range(ny1, ny2 + 1):
        f = 1 - float((ny - ny1 + 1)) / (ny2 - ny1 + 1)
        nx = int(np.round(f * w / dx))
        nx1 = int(np.floor((num_x - nx) / 2.0))
        nx2 = nx1 + nx - 1
        EPS_DEV_1[nx1:nx2, ny - 1] = eps_ref
        U_DEV_1[nx1:nx2, ny - 1] = u_ref

    # ----- Step 3: Compute Convolution Matrices ------------------------------------------
    EPS_DEV_1_C = convmat2D(EPS_DEV_1, num_harmonics_x, num_harmonics_y)
    U_DEV_1_C = convmat2D(U_DEV_1, num_harmonics_x, num_harmonics_y)

    EPS_DEV_2_C = convmat2D(EPS_DEV_2, num_harmonics_x, num_harmonics_y)
    U_DEV_2_C = convmat2D(U_DEV_2, num_harmonics_x, num_harmonics_y)

    # ----- Step 4: Compute Wave Vector Expansion, and normalize with respect to k0 -------
    k_inc_norm = np.sqrt(u_ref * eps_ref) * np.array([np.sin(np.deg2rad(theta))*np.cos(np.deg2rad(psi)),
                                                      np.sin(np.deg2rad(theta))*np.sin(np.deg2rad(psi)),
                                                      np.cos(np.deg2rad(theta))
                                                      ]
                                                     )

    # Note that in python 2, 3/2 = 1, but in python 3, 3/2 = 1.5 (automatic type conversion)
    kx_norm = k_inc_norm[0] - 2*np.pi*np.arange(-int(num_harmonics_x/2),
                                                int(num_harmonics_x/2) + 1)/(k0*Lx*unit)
    ky_norm = k_inc_norm[1] - 2*np.pi*np.arange(-int(num_harmonics_y/2),
                                                int(num_harmonics_y/2) + 1)/(k0*Ly*unit)

    # KX = kx1, ..., kxm, kx1, ..., kxm, kx1, ..., kxm repeated N times, total MN diagonal
    KX = dia_matrix((np.array(kx_norm.tolist() * num_harmonics_y), [0]), shape=(MN, MN))
    # KX = dia_matrix((kx_norm.repeat(num_harmonics_y), [0]), shape=(MN, MN))

    # KY = ky1 ky1 ... M times ... ky2, ky2, ... M times ... kyn, kyn, ... M times, total MN diagonal
    KY = dia_matrix((ky_norm.repeat(num_harmonics_x), [0]), shape=(MN, MN))
    # KY = dia_matrix((np.array(ky_norm.tolist() * num_harmonics_x), [0]), shape=(MN, MN))

    I = dia_matrix((np.ones(MN), [0]), shape=(MN, MN))
    kz_ref = -np.conj(sc.sqrt(u_ref*eps_ref*I.diagonal() - KX.diagonal()**2 - KY.diagonal()**2))
    kz_trn = np.conj(sc.sqrt(u_trn*eps_trn*I.diagonal() - KX.diagonal()**2 - KY.diagonal()**2))
    del I
    KZ_REF = dia_matrix((kz_ref, [0]), shape=(MN, MN))
    KZ_TRN = dia_matrix((kz_trn, [0]), shape=(MN, MN))

    # ----- Step 5: Compute Eigen-modes of free space ----------------------------------------------
    FreeSpace = rcwa.HomoLayer(num_harmonics_x, num_harmonics_y,
                               Lx, Ly, d=0,
                               u_r=1, eps_r=1, KX=KX, KY=KY,
                               units=unit)

    W0 = FreeSpace.cal_W()
    V0 = FreeSpace.cal_V()

    # ----- Step 6: Initialize Device Scattering Matrix --------------------------------------------
    I = dia_matrix((np.ones(2*MN), [0]), shape=(2*MN, 2*MN)).toarray()
    Z = dia_matrix((np.zeros(2*MN), [0]), shape=(2*MN, 2*MN)).toarray()

    S11_dev = Z
    S12_dev = I
    S21_dev = S12_dev
    S22_dev = S11_dev

    # ----- Step 7: Main loop iterates through layers ----------------------------------------------
    # Initialize parameters
    InitialVal = rcwa.Initial(u_ref, eps_ref,
                              u_trn, eps_ref,
                              theta, psi,
                              lambda_0,
                              unit,
                              pol
                              )

    # Layers
    LayerOne = rcwa.Layer(num_harmonics_x, num_harmonics_y,
                          Lx, Ly, d1,
                          U_DEV_1_C, EPS_DEV_1_C,
                          units=unit)

    LayerTwo = rcwa.Layer(num_harmonics_x, num_harmonics_y,
                          Lx, Ly, d2,
                          U_DEV_2_C, EPS_DEV_2_C,
                          units=unit)

    # Compute layer scattering matrix for the i-th layer
    S11_1, S12_1, S21_1, S22_1 = LayerOne.cal_S_param(KX, KY, W0, V0, lambda_0)
    S11_2, S12_2, S21_2, S22_2 = LayerTwo.cal_S_param(KX, KY, W0, V0, lambda_0)

    # Update device scattering matrix
    S11_dev, S12_dev, S21_dev, S22_dev = rcwa.redheffer_std([[S11_dev, S12_dev],[S21_dev, S22_dev]],
                                                            [[S11_1, S12_1], [S21_1, S22_1]])
    S11_dev, S12_dev, S21_dev, S22_dev = rcwa.redheffer_std([[S11_dev, S12_dev], [S21_dev, S22_dev]],
                                                            [[S11_2, S12_2], [S21_2, S22_2]])

    # ----- Step 8: Compute Reflection side connection S-matrix
    Region_ref = rcwa.RegionLayer(num_harmonics_x, num_harmonics_y,
                                  Lx, Ly, d=0,
                                  u_r=u_ref, eps_r=eps_ref,
                                  KX=KX, KY=KY, KZ=KZ_REF,
                                  region='ref', units=unit)
    S11_ref, S12_ref, S21_ref, S22_ref = Region_ref.cal_S(W0, V0)

    # ----- Step 9: Compute Transmission side connection S-matrix
    Region_trn = rcwa.RegionLayer(num_harmonics_x, num_harmonics_y,
                                  Lx, Ly, d=0,
                                  u_r=u_trn, eps_r=eps_trn,
                                  KX=KX, KY=KY, KZ=KZ_TRN,
                                  region='trn', units=unit)
    S11_trn, S12_trn, S21_trn, S22_trn = Region_trn.cal_S(W0, V0)

    # ----- Step 10: Compute global Scattering Matrix
    S11_g, S12_g, S21_g, S22_g = rcwa.redheffer_std([[S11_ref, S12_ref], [S21_ref, S22_ref]],
                                                    [[S11_dev, S12_dev], [S21_dev, S22_dev]])
    S11_g, S12_g, S21_g, S22_g = rcwa.redheffer_std([[S11_g, S12_g], [S21_g, S22_g]],
                                                    [[S11_trn, S12_trn], [S21_trn, S22_trn]])

    # ----- Step 11: Compute reflected and transmitted fields
    # Compute Source Field
    e_src = InitialVal.get_source_field(num_harmonics_x, num_harmonics_y)
    Wref = Region_ref.cal_W()
    Wtrn = Region_trn.cal_W()

    # Compute Transmission and Reflection Modal Coefficients
    rx, ry, rz = rcwa.get_ref_field(num_harmonics_x, num_harmonics_y, e_src, Wref, S11_g, KX, KY, KZ_REF)
    tx, ty, tz = rcwa.get_trn_field(num_harmonics_x, num_harmonics_y, e_src, Wref, Wtrn, S21_g, KX, KY, KZ_TRN)

    # ----- Step 12: Compute diffraction efficiencies
    R = rcwa.get_R(rx, ry, rz, KZ_REF, k_inc_norm)
    T = rcwa.get_T(tx, ty, tz, KZ_TRN, k_inc_norm, u_ref, u_trn)

    R_tot = np.sum(R) * 100
    T_tot = np.sum(T) * 100

    # ----- Step 13: Verify conservation of energy
    print('R = %.3f %%' % R_tot)
    print('T = %.3f %%' % T_tot)
    print('------------------')
    print('CON = %.2f %%' % (R_tot + T_tot))

    # ----- Step Appendix: plot the device's eps_r, u_r and convolution matrices -------------------
    # plt.subplot(221)
    # plt.pcolormesh(xa, ya, EPS_DEV_1.T, cmap='RdBu_r')
    # plt.axis('equal')
    # plt.title('UNIT CELL ($\epsilon_r$)')
    # plt.colorbar()
    #
    # plt.subplot(222)
    # plt.pcolormesh(xa, ya, U_DEV_1.T, cmap='RdBu_r')
    # plt.axis('equal')
    # plt.title('UNIT CELL ($\mu_r$)')
    # plt.colorbar()
    #
    # plt.subplot(223)
    # plt.imshow(np.real(EPS_DEV_1_C.T), interpolation='none', cmap='jet')
    # plt.axis('equal')
    # plt.title('$\epsilon_r$ CONVOLUTION MATRIX')
    # plt.colorbar()
    #
    # plt.subplot(224)
    # plt.imshow(np.real(U_DEV_1_C.T), interpolation='none', cmap='jet')
    # plt.axis('equal')
    # plt.title('$\mu_r$ CONVOLUTION MATRIX')
    # plt.colorbar()
    # plt.show()


