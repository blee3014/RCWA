import numpy as np
from numpy.linalg import inv, eig
import scipy as sc
from scipy.linalg import expm
from scipy.sparse import dia_matrix


# Todo: functions for computing stuff
def redheffer_std(Layer1, Layer2):
    pass


def redheffer_rev(Layer1, Layer2):
    pass


def get_ref_field():
    pass


def get_trn_field():
    pass


def get_long_comp():
    pass


def get_R():
    pass


def get_T():
    pass


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
        :return: 2MN column vectore that excites the eigenmodes
        in the layers in RCWA simulation based on the given polarization
        """
        pol = self.get_polar_vector()
        delta = np.array([np.hstack((np.zeros(int(np.floor(M*N/2))), np.array([1]), np.zeros(int(np.floor(M*N/2)))))]).T
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
        self._Lx = Lx
        self._Ly = Ly
        self._d = d
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

    def _cal_P(self, KX, KY):
        A = inv(self._eps)
        P11 = KX @ A @ KY
        P12 = self._u - KX @ A @ KX
        P21 = KY @ A @ KY - self._u
        P22 = -KY @ A @ KX
        del A
        return np.bmat([[P11, P12], [P21, P22]])

    def _cal_Q(self, KX, KY):
        A = inv(self._u)
        Q11 = KX @ A @ KY
        Q12 = self._eps - KX @ A @ KX
        Q21 = KY @ A @ KY - self._eps
        Q22 = -KY @ A @ KX
        del A
        return np.bmat([[Q11, Q12], [Q21, Q22]])

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
        P = self._cal_P(KX, KY)
        Q = self._cal_Q(KX, KY)
        OMEGA2 = P @ Q
        del P
        LAM2, W = eig(OMEGA2)
        LAM = sc.sqrt(LAM2)
        del OMEGA2, LAM2
        LAM = dia_matrix((LAM, [0]), shape=(mat_size, mat_size))
        V = Q @ W @ inv(LAM.toarray())
        del Q
        return W, V, LAM

    def cal_S_param(self, W, V, LAM, W0, V0, wavelength):
        """
        Having solved the Matrix wave equation with "cal_eig()", calculate the S parameters for the scattering matrix.
        :param W: Eigenvector matrix for E field (numpy.ndarray)
        :param V: Eigenvector matrix for H field (numpy.ndarray)
        :param LAM: Eigenvalues diagonal matrix (scipy.sparse.dia.dia_matrix)
        :param W0: Eigenvector matrix from free space padding (see Lecture 21 from http://emlab.utep.edu/ee5390cem.htm)
        :param V0: Eigenvector matrix from free space padding (see Lecture 21 from http://emlab.utep.edu/ee5390cem.htm)
        :param wavelength: Incident wavelength. It is multiplied by the units from get_units() method.
        :return: S11, S12, S21, S22
        """
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

    # Todo: calculate longitudinal K vectors
    def cal_KZ(self, KX, KY):
        pass


# Todo: class for homogeneous layer
class HomoLayer(Layer):
    """
    Class HomoLayer has homogeneous layer information for RCWA simulation.
    M     = Number of spatial harmonics in the x direction
    N     = Number of spatial harmonics in the y direction
    Lx    = Size of the device unit cell in the x direction
    Ly    = Size of the device unit cell in the y direction
    d     = Thickness of the layer
    u_r   = Relative permeability scalar of the layer
    eps_r = Relative permittivity scalar of the layer

    Uses the same __init__ method except u and eps are scalars rather than
    convolution matrices.
    """
    def _cal_P(self, KX, KY):
        pass

    def _cal_Q(self, KX, KY):
        pass

    def _cal_LAM(self):
        pass

    def cal_W(self):
        pass

    def cal_V(self):
        pass


class RegionLayer(HomoLayer):
    def cal_S(self, W0, V0):
        A = []
        B = []
        S11 = []
        S12 = []
        S21 = []
        S22 = []
        return S11, S12, S21, S22

