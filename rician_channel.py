from numpy.random import default_rng
import numpy as np
from scipy.constants import speed_of_light
import math

rng = default_rng()


class RicianChannel:

    def __init__(self, nr, nt, alpha, beta, d, c0, fc, delta_r=0.5,
                 delta_t=0.5, phi_r=0, phi_t=0, a=1):
        self.nr = nr
        self.nt = nt
        self.alpha = alpha
        self.beta = beta
        self.d = d
        self.c0 = c0
        self.lambdac = speed_of_light / fc
        self.delta_r = delta_r
        self.delta_t = delta_t
        self.phi_r = phi_r
        self.phi_t = phi_t
        self.a = a
        self.H = None

    def get_channel_matrix(self):
        if self.H is None:
            if math.isinf(self.beta):
                los = 1
                nlos = 0
            else:
                los = np.sqrt(self.beta / (1+self.beta))
                nlos = np.sqrt(1 / (1+self.beta))
            e_r = np.exp(-1j * 2 * np.pi * np.arange(self.nr) *
                         self.delta_r * np.cos(self.phi_r))
            e_r = e_r.reshape((e_r.shape[0], 1))
            e_t = np.exp(-1j * 2 * np.pi * np.arange(self.nt) *
                         self.delta_r * np.cos(self.phi_t))
            e_t = e_t.reshape((e_t.shape[0], 1))
            H_los = self.a * np.sqrt(self.nr * self.nt) * np \
                .exp(-1j * 2 * np.pi * self.d / self.lambdac) * e_r * e_t.conj().T
            H_nlos = (1 / np.sqrt(2)) * (rng.standard_normal((self.nr, self.nt))
                                         + 1j * rng.standard_normal((self.nr, self.nt)))
            self.H = np.sqrt(self.c0*np.power(self.d, -self.alpha))*(los * H_los + nlos * H_nlos)
        return self.H
