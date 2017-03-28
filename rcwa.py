import numpy as np
from numpy.linalg import inv, eig
import scipy as sc
from scipy.linalg import expm
from scipy.sparse import dia_matrix


def redheffer_std(S_global, S_i):
    """
    S_global = S_global X S_i
    :param S_global: S parameters of the global scattering matrix
    :param S_i: S parameters of the additional scattering matrix
    :return: new updated S_global S parameters, S11, S12, S21, S22
    """
    MN, _ = S_global[0][0].shape
    I = dia_matrix((np.ones(MN), [0]), shape=(MN, MN))
    D = S_global[0][1] @ inv(I - S_i[0][0] @ S_global[1][1])
    F = S_i[1][0] @ inv(I - S_global[1][1] @ S_i[0][0])
    S11 = S_global[0][0] + D @ S_i[0][0] @ S_global[1][0]
    S12 = D @ S_i[0][1]
    S21 = F @ S_global[1][0]
    S22 = S_i[1][1] + F @ S_global[1][1] @ S_i[0][1]

    return S11, S12, S21, S22


def redheffer_rev(S_global, S_i):
    """
    S_global = S_i X S_global
    :param S_global: S parameters of the global scattering matrix
    :param S_i: S parameters of the additional scattering matrix
    :return: new updated S_global S parameters, S11, S12, S21, S22
    """
    MN, _ = S_global[0].shape
    I = dia_matrix((np.ones(MN), [0]), shape=(MN, MN))
    D = S_i[0][1] @ inv(I - S_global[0][0] @ S_i[1][1])
    F = S_global[1][0] @ inv(I - S_i[1][1] @ S_global[0][0])
    S22 = S_global[1][1] + F @ S_i[1][1] @ S_global[0][1]
    S21 = F @ S_i[1][0]
    S12 = D @ S_global[0][1]
    S11 = S_i[0][0] + D @ S_global[0][0] @ S_i[1][0]

    return S11, S12, S21, S22


def get_ref_field(M, N, e_src, Wref, S11_g, KX, KY, KZ_REF):
    MN = M*N
    # Compute Source Modal Coefficients
    c_src = inv(Wref) @ e_src
    c_ref = S11_g @ c_src
    e_ref = Wref @ c_ref
    rx = e_ref[0:MN]
    ry = e_ref[MN:2 * MN]
    # Compute Longitudinal Components
    rz = -inv(KZ_REF.toarray()) @ (KX @ rx + KY @ ry)
    return rx, ry, rz


def get_trn_field(M, N, e_src, Wref, Wtrn, S21_g, KX, KY, KZ_TRN):
    MN = M*N
    # Compute Source Modal Coefficients
    c_src = inv(Wref) @ e_src
    c_trn = S21_g @ c_src
    e_trn = Wtrn @ c_trn
    tx = e_trn[0:MN]
    ty = e_trn[MN:2 * MN]
    # Compute Longitudinal Components
    tz = -inv(KZ_TRN.toarray()) @ (KX @ tx + KY @ ty)
    return tx, ty, tz


def get_R(rx, ry, rz, KZ_REF, k_inc_norm):
    r_2 = abs(rx.getA()) ** 2 + abs(ry.getA()) ** 2 + abs(rz.getA()) ** 2
    R = np.real(np.dot(-KZ_REF.toarray()/k_inc_norm[2], r_2))
    return R


def get_T(tx, ty, tz, KZ_TRN, k_inc_norm, u_ref, u_trn):
    t_2 = abs(tx.getA()) ** 2 + abs(ty.getA()) ** 2 + abs(tz.getA()) ** 2
    T = np.real(np.dot((u_ref / u_trn) * KZ_TRN.toarray() / k_inc_norm[2], t_2))
    return T


class Initial(object):
    """
    Class with Initial values of the simulation.
    (u_ref, eps_ref)     = u and eps of Region I (reflection region)
    (u_trn, eps_trn)     = u and eps of Region II (transmission region)
    (theta_inc, psi_inc) = incident angles. 'theta' is angle between incident k vector and z axis
                                            'psi' is polar angle in the xy plane of the incident beam
    wavelength           = wavelength of the incident beam
    units                = wavelength units, e.g. um = 1e-6
    polarization         = polarization vector in terms of (TE, TM). E.g., TE polarization is (1, 0)
                           and half TE half TM is (0.5, 0.5). REQUIREMENT: TE + TM = 1
    """
    def __init__(self,
                 u_ref, eps_ref,
                 u_trn, eps_trn,
                 theta_inc, psi_inc,
                 wavelength,
                 units=1e-6,
                 polarization=(0, 1)
                 ):
        self._u_ref, self._eps_ref = (u_ref, eps_ref)
        self._u_trn, self._eps_trn = (u_trn, eps_trn)
        self._theta_inc, self._psi_inc = np.deg2rad([theta_inc, psi_inc])
        self._units = units
        self._lamb0 = wavelength*self._units
        self._k0 = 2*np.pi/self._lamb0
        self._p_TE = polarization[0]
        self._p_TM = polarization[1]

        if self._p_TE + self._p_TM != 1.0:
            raise ValueError("Polarization ratio must add up to 1 (TE + TM = 1).")

    # getter and setter methods
    def get_u_ref(self):
        return self._u_ref

    def get_eps_ref(self):
        return self._eps_ref

    def get_u_trn(self):
        return self._u_trn

    def get_eps_trn(self):
        return self._eps_trn

    def get_theta(self):
        return self._theta_inc

    def get_psi(self):
        return self._psi_inc

    def get_wavelength(self):
        return self._lamb0

    def get_units(self):
        return self._units

    def set_u_ref(self, u_ref):
        self._u_ref = u_ref

    def set_eps_ref(self, eps_ref):
        self._eps_ref = eps_ref

    def set_u_trn(self, u_trn):
        self._u_trn = u_trn

    def set_eps_trn(self, eps_trn):
        self._eps_trn = eps_trn

    def set_theta(self, theta_inc):
        self._theta_inc = theta_inc

    def set_psi(self, psi_inc):
        self._psi_inc = psi_inc

    def set_wavelength(self, wavelength, update_k0=True):
        self._lamb0 = wavelength*self._units
        if update_k0:
            self._k0 = 2*np.pi/self._lamb0

    def set_units(self, units):
        self._units = units

    def _get_te_unit(self):
        """
        Returns TE unit vector from theta_inc and psi_inc
        :return: TE unit vector [TE_x, TE_y, TE_z]
        """
        if self._theta_inc != 0:
            TE = np.array([-np.sin(self._psi_inc),
                           np.cos(self._psi_inc),
                           0.0])
        else:
            TE = np.array([0.0, 1.0, 0.0])

        return TE

    def _get_tm_unit(self):
        """
        Returns TM unit vector from theta_inc and psi_inc
        :return: TM unit vector [TM_x, TM_y, TM_z]
        """
        if self._theta_inc != 0:
            TM = np.array([np.cos(self._psi_inc) * np.cos(self._theta_inc),
                           -np.sin(self._psi_inc) * np.cos(self._theta_inc),
                           -np.sin(self._theta_inc)])
        else:
            TM = np.array([1.0, 0.0, 0.0])

        return TM

    def get_polar_vector(self):
        """
        :return: Polarization vector numpy array [x, y, z] with unit length
        """
        te_unit = self._get_te_unit()
        tm_unit = self._get_tm_unit()
        pol_init = self._p_TE * te_unit + self._p_TM * tm_unit
        pol_mag = np.sqrt((pol_init**2).sum())
        return pol_init/pol_mag

    def get_source_field(self, M, N):
        """
        :return: 2MN column vector that excites the Eigen-modes
        in the layers in RCWA simulation based on the given polarization
        """
        pol = self.get_polar_vector()
        delta = np.array([np.hstack((np.zeros(int(M*N/2)), np.array([1]), np.zeros(int(M*N/2))))]).T
        return np.vstack((pol[0]*delta, pol[1]*delta))


class Layer(object):
    """
    Class Layer is used to create a periodic layer for the RCWA simulation.
    M     = Number of spatial harmonics in the x direction
    N     = Number of spatial harmonics in the y direction
    Lx    = Size of the device unit cell in the x direction
    Ly    = Size of the device unit cell in the y direction
    d     = Thickness of the layer
    u_r   = Relative permeability convolution matrix of the unit cell
    eps_r = Relative permittivity convolution matrix of the unit cell
    """
    def __init__(self, M, N, Lx, Ly, d, u_r, eps_r, units=1e-6):
        self._M = M
        self._N = N
        self._Lx = Lx * units
        self._Ly = Ly * units
        self._d = d * units
        self._u = u_r
        self._eps = eps_r
        self._units = units

    # getter and setter methods
    def get_M(self):
        return self._M

    def get_N(self):
        return self._N

    def get_Lx(self):
        return self._Lx

    def get_Ly(self):
        return self._Ly

    def get_d(self):
        return self._d

    def get_u(self):
        return self._u

    def get_eps(self):
        return self._eps

    def get_units(self):
        return self._units

    def set_M(self, M):
        self._M = M

    def set_N(self, N):
        self._N = N

    def set_Lx(self, Lx):
        self._Lx = Lx

    def set_Ly(self, Ly):
        self._Ly = Ly

    def set_d(self, d):
        self._d = d

    def set_u(self, u):
        self._u = u

    def set_eps(self, eps):
        self._eps = eps

    def set_units(self, units):
        self._units = units

    def cal_P(self, KX, KY):
        A = inv(self._eps)
        P11 = KX @ A @ KY
        P12 = self._u - KX @ A @ KX
        P21 = KY @ A @ KY - self._u
        P22 = -KY @ A @ KX

        # P11[abs(np.imag(P11)) < np.finfo(np.float).eps] = np.real(P11[abs(np.imag(P11)) < np.finfo(np.float).eps]) + 0 * 1j
        # P12[abs(np.imag(P12)) < np.finfo(np.float).eps] = np.real(P12[abs(np.imag(P12)) < np.finfo(np.float).eps]) + 0 * 1j
        # P21[abs(np.imag(P21)) < np.finfo(np.float).eps] = np.real(P21[abs(np.imag(P21)) < np.finfo(np.float).eps]) + 0 * 1j
        # P22[abs(np.imag(P22)) < np.finfo(np.float).eps] = np.real(P22[abs(np.imag(P22)) < np.finfo(np.float).eps]) + 0 * 1j

        del A
        return np.bmat([[P11, P12],
                        [P21, P22]])

    def cal_Q(self, KX, KY):
        A = inv(self._u)
        Q11 = KX @ A @ KY
        Q12 = self._eps - KX @ A @ KX
        Q21 = KY @ A @ KY - self._eps
        Q22 = -KY @ A @ KX
        del A
        return np.bmat([[Q11, Q12],
                        [Q21, Q22]])

    def cal_eig(self, KX, KY):
        """
        Solves the Matrix wave equation for the layer.
        :param KX: MN x MN matrix with diagonal elements as kx1, kx1, ... M times ... , kx2, kx2, ... M times ... etc.
        :param KY: MN x MN matrix with diagonal elements as ky1, ky2, ... kyn, ky1, ky2, ... kyn, ky1, ky2, ... etc.
        :return: W   = Eigenvector matrix for E field (numpy.ndarray)
                 V   = Eigenvector Matrix for H field (numpy.ndarray)
                 LAM = Eigenvalues (scipy.sparse.dia.dia_matrix)
        """
        mat_size = 2*self._M*self._N
        P = self.cal_P(KX, KY)
        Q = self.cal_Q(KX, KY)
        OMEGA2 = P @ Q
        del P
        LAM2, W = eig(OMEGA2)
        LAM = sc.sqrt(LAM2)
        del OMEGA2, LAM2
        LAM = dia_matrix((LAM, [0]), shape=(mat_size, mat_size))
        V = Q @ W @ inv(LAM.toarray())
        del Q
        return W, LAM, V

    def cal_S_param(self, KX, KY, W0, V0, wavelength):
        """
        Having solved the Matrix wave equation with "cal_eig()", calculate the S parameters for the scattering matrix.
        :param KX: Diagonal matrix with kx_norm(m,n) on its diagonal
        :param KY: Diagonal matrix with ky_norm(m,n) on its diagonal
        :param W0: Eigenvector matrix from free space padding (see Lecture 21 from http://emlab.utep.edu/ee5390cem.htm)
        :param V0: Eigenvector matrix from free space padding (see Lecture 21 from http://emlab.utep.edu/ee5390cem.htm)
        :param wavelength: Incident wavelength. It is multiplied by the units from get_units() method.
        :return: S11, S12, S21, S22
        """
        W, LAM, V = self.cal_eig(KX, KY)
        k0 = 2*np.pi/(wavelength*self.get_units())
        Ai0 = inv(W) @ W0 + inv(V) @ V0
        Bi0 = inv(W) @ W0 - inv(V) @ V0
        Xi = expm(-LAM.toarray() * k0 * self.get_d())
        D = inv(Ai0 - Xi @ Bi0 @ inv(Ai0) @ Xi @ Bi0)
        S11 = D @ (Xi @ Bi0 @ inv(Ai0) @ Xi @ Ai0 - Bi0)
        S12 = D @ Xi @ (Ai0 - Bi0 @ inv(Ai0) @ Bi0)
        del Ai0, Bi0, Xi, D, k0
        S21 = S12
        S22 = S11
        return S11, S12, S21, S22


class HomoLayer(Layer):
    """
    Inherits from Layer Class.
    Class HomoLayer has homogeneous layer information for RCWA simulation.
    M     = Number of spatial harmonics in the x direction
    N     = Number of spatial harmonics in the y direction
    Lx    = Size of the device unit cell in the x direction
    Ly    = Size of the device unit cell in the y direction
    d     = Thickness of the layer
    u_r   = Relative permeability scalar of the layer
    eps_r = Relative permittivity scalar of the layer
    KX    = Diagonal matrix with kx(M,N) on the diagonals
    KY    = Diagonal matrix with ky(M,N) on the diagonals
    units = units to be used with wavelength, distance, etc
    
    Note: u and eps are scalars rather than convolution matrices.
    """
    def __init__(self, M, N, Lx, Ly, d, u_r, eps_r, KX, KY, units=1e-6):
        super().__init__(M, N, Lx, Ly, d, u_r, eps_r, units)
        self._I = dia_matrix((np.ones(M*N), [0]), shape=(M*N, M*N))
        self._Kz = self.get_Kz(KX, KY)
        self._0 = dia_matrix((np.zeros(M*N), [0]), shape=(M*N, M*N))
        self._KX = KX
        self._KY = KY

    def get_Kz(self, KX, KY):
        return np.conj(sc.sqrt(self._I.toarray() - KX**2 - KY**2))

    def cal_Q(self):
        KX = self._KX
        KY = self._KY
        u = self.get_u()
        eps = self.get_eps()
        Q11 = KX @ KY
        Q12 = u*eps*self._I - KX**2
        Q21 = KY**2 - u*eps*self._I
        Q22 = -KX @ KY
        del KX, KY, eps
        return np.bmat([[Q11.toarray(), Q12.toarray()],
                     [Q21.toarray(), Q22.toarray()]])/u

    def cal_LAM(self):
        return np.bmat([[1j*self._Kz, self._0.toarray()],
                        [self._0.toarray(), 1j*self._Kz]])

    def cal_W(self):
        return np.bmat([[self._I.toarray(), self._0.toarray()],
                        [self._0.toarray(), self._I.toarray()]])

    def cal_V(self):
        LAM = self.cal_LAM()
        Q = self.cal_Q()
        return Q @ inv(LAM)


class RegionLayer(HomoLayer):
    """
    Inherits from HomoLayer Class.
    Class HomoLayer has homogeneous layer information for RCWA simulation.
    M     = Number of spatial harmonics in the x direction
    N     = Number of spatial harmonics in the y direction
    Lx    = Size of the device unit cell in the x direction
    Ly    = Size of the device unit cell in the y direction
    d     = Thickness of the layer
    u_r   = Relative permeability scalar of the layer
    eps_r = Relative permittivity scalar of the layer
    KX    = Diagonal matrix with kx(M,N) on the diagonals
    KY    = Diagonal matrix with ky(M,N) on the diagonals
    KZ    = Diagonal matrix with kz_ref(M,N) or kz_trn(M,N) on the diagonals
    region= string, either 'ref' or 'trn' for reflection, transmission region, respectively.
    units = units to be used with wavelength, distance, etc
    
    """
    def __init__(self, M, N, Lx, Ly, d, u_r, eps_r, KX, KY, KZ, region, units=1e-6):
        super().__init__(M, N, Lx, Ly, d, u_r, eps_r, KX, KY, units)
        self._region = region
        self._Kz = KZ.toarray()

    def cal_LAM(self):
        if self._region == 'ref':
            return np.bmat([[-1j*self._Kz, self._0.toarray()],
                            [self._0.toarray(), -1j*self._Kz]])
        if self._region == 'trn':
            return np.bmat([[1j * self._Kz, self._0.toarray()],
                            [self._0.toarray(), 1j * self._Kz]])
    def cal_S(self, W0, V0):
        Wref = self.cal_W()
        Vref = self.cal_V()
        A = inv(W0) @ Wref + inv(V0) @ Vref
        B = inv(W0) @ Wref - inv(V0) @ Vref
        del Wref, Vref
        inverse = inv(A)

        # Reflection side
        if self._region == 'ref':
            S11 = -inverse @ B
            S12 = 2*inverse
            S21 = 0.5*(A - B @ inverse @ B)
            S22 = B @ inverse
            del inverse

        # Transmission side
        elif self._region == 'trn':
            S11 = B @ inverse
            S12 = 0.5 * (A - B @ inverse @ B)
            S21 = 2 * inverse
            S22 = - inverse @ B
            del inverse

        return S11, S12, S21, S22

