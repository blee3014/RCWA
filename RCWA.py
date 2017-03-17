import numpy as np
import matplotlib.pyplot as plt
from conv import convmat2D
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
    (u_ref, eps_ref) = u and eps of Region I (reflection region)
    (u_trn, eps_trn) = u and eps of Region II (transmission region)
    (theta_inc, psi_inc) = incident angles. theta is angle between k vector and z axis
                                            psi is polar angle in the xy plane of the incident beam
    wavelength = wavelength of the incident beam
    units = wavelength units, e.g. um = 1e-6
    polarization = polarization vector in terms of (TE, TM). E.g., TE polarization is (1, 0)
                   and half TE half TM is (0.5, 0.5). TE + TM = 1
    """
    def __init__(self,
                 (u_ref, eps_ref),
                 (u_trn, eps_trn),
                 (theta_inc, psi_inc),
                 wavelength,
                 units=1e-6,
                 polarization=[0,1]
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

    def get_u_ref(self):
        return self._u_ref

    def get_eps_ref(self):
        return self._eps_ref

    def get_u_trn(self):
        return self._u_trn

    def get_eps_trn(self):
        return self._eps_trn

    def get_theta_inc(self):
        return self._theta_inc

    def get_psi_inc(self):
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

    def set_theta_inc(self, theta_inc):
        self._theta_inc = theta_inc

    def set_psi_inc(self, psi_inc):
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
        te_unit = self._get_te_unit()
        tm_unit = self._get_tm_unit()
        pol_init = self._p_TE * te_unit + self._p_TM * tm_unit
        pol_mag = np.sqrt((pol_init**2).sum())
        return pol_init/pol_mag

    def get_source_field(self, M, N):
        pol = self.get_polar_vector()
        delta = np.array([np.hstack((np.zeros(M*N/2), np.array([1]), np.zeros(M*N/2)))]).T
        return np.vstack((pol[0]*delta, pol[1]*delta))


# Todo: class for Layer
class Layer(object):
    def __init__(self):
        pass


# Todo: class for reflection region
class RegionI(Layer):
    pass


# Todo: class for Transmission region
class RegionII(Layer):
    pass
