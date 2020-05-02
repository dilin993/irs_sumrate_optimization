from rician_channel import RicianChannel
import util
import numpy as np
from nodes import BS, IRS, UE
import math
from scipy import optimize


class Simulation:

    def simulate(self):
        pass

    def result(self):
        pass


def get_cost(v, a, b, num_users, N0, gamma):
    cost = 0
    for k in range(num_users):
        psig = np.power(np.abs(np.matmul(v.conj().T, a[(k, k)]) + b[(k, k)]), 2)
        pint = 0
        for j in range(num_users):
            if k != j:
                pint = pint + np.power(np.abs(np.matmul(v.conj().T, a[(j, k)]) + b[(j, k)]), 2)
        cost = cost - np.log2(1+psig/(pint+N0))
    return cost + np.sum(gamma*np.abs((np.abs(v)-1)))


def get_gradient(v, a, b, num_users, N0, gamma):
    grad = np.zeros(v.shape, dtype=complex)
    for k in range(num_users):
        p1 = np.zeros(v.shape, dtype=complex)
        p2 = np.zeros(v.shape, dtype=complex)
        norm1 = 0
        norm2 = 0
        for j in range(num_users):
            dp = np.matmul(a[(j, k)], a[(j, k)].conj().T) * v +\
                 a[(j, k)] * b[(j, k)].conj().T
            dnorm = np.power(np.abs(np.matmul(v.conj().T, a[(j, k)]) + b[(j, k)]), 2)
            p1 = p1 + dp
            norm1 = norm1 + dnorm
            if k != j:
                p2 = p2 + dp
                norm2 = norm2 + dnorm
        grad = grad - 2*((p1/(norm1 + N0))-(p2/(norm2+N0)))
    return grad + gamma*(np.divide(v, np.abs(v)))


def real_to_complex(z):      # real vector of length 2n -> complex of length n
    return z[:len(z)//2] + 1j * z[len(z)//2:]


def complex_to_real(z):      # complex vector of length n -> real of length 2n
    return np.concatenate((np.real(z), np.imag(z)))


def get_cost_real(v, a, b, num_users, N0, gamma):
    return get_cost(real_to_complex(v), a, b, num_users, N0, gamma)


def get_gradient_real(v, a, b, num_users, N0, gamma):
    return complex_to_real(get_gradient(real_to_complex(v), a, b, num_users, N0, gamma))


class MUIRSSimuation(Simulation):

    def __init__(self, fc, antnum_bs, pos_bs, n_irs, pos_irs, sigma_sqr_dB,
                 SNR_dB, c0_dB, alpha_bs_irs, beta_bs_irs, gamma, tol=1e-2, maxiter=100):
        self.fc = fc
        self.bs = BS(antnum_bs, pos_bs)
        self.irs = IRS(n_irs, pos_irs)
        self.sigma_sqr = util.db2lin(sigma_sqr_dB)
        self.SNR_dB = SNR_dB
        self.c0 = util.db2lin(c0_dB)
        self.bs_irs_link = RicianChannel(n_irs, antnum_bs, alpha_bs_irs,
                                         beta_bs_irs, util.distance(self.bs, self.irs), self.c0, fc)
        self.gamma = gamma
        self.tol = tol
        self.maxiter = maxiter
        self.users = []
        self.bs_user_links = []
        self.irs_user_links = []
        self.phi = None

    def add_user(self, pos, alpha_bs_u, alpha_irs_u, beta_bs_u, beta_irs_u):
        idx = len(self.users)
        self.users.append(UE(1, pos))
        self.bs_user_links.append(RicianChannel(1, self.bs.antnum,
                                                alpha_bs_u, beta_bs_u,
                                                util.distance(self.bs, self.users[idx]),
                                                self.c0, self.fc))
        self.irs_user_links.append(RicianChannel(1, self.irs.antnum,
                                                 alpha_irs_u, beta_irs_u,
                                                 util.distance(self.irs, self.users[idx]),
                                                 self.c0, self.fc))

    def num_users(self):
        return len(self.users)

    def simulate(self):
        p_irs = np.zeros(self.SNR_dB.shape)
        p_0 = np.zeros(self.SNR_dB.shape)
        r_irs = np.zeros(self.SNR_dB.shape)
        r_0 = np.zeros(self.SNR_dB.shape)
        phi = self.get_phi()
        idx = 0
        for SNR in self.SNR_dB:
            SNR = util.db2lin(SNR)
            print('############################################')
            print('Simulating SNR: ', SNR)
            H = np.zeros((self.num_users(), self.bs.antnum), dtype=complex)
            for k in range(self.num_users()):
                H[:, k] = self.bs_user_links[k].get_channel_matrix()
            Wopt = self.get_mmse(H, SNR)
            p_0[idx] = util.lin2dbm(np.sum(np.power(np.abs(Wopt), 2)))
            r_0[idx] = self.get_sum_rate(Wopt)
            vopt, Wopt_irs = self.optimize(SNR)
            p_irs[idx] = util.lin2dbm(np.sum(np.power(np.abs(Wopt_irs), 2)))
            r_irs[idx] = self.get_sum_rate(Wopt_irs, vopt)
            idx = idx + 1
        results = [SimulationResult('BS Power Consumption', 'SNR[dB]', 'Power[dBm]'),
                   SimulationResult('Sum rate', 'SNR(dB)', 'Rate[bits/s/HZ]')]
        results[0].add_result(self.SNR_dB, p_irs, 'With IRS')
        results[0].add_result(self.SNR_dB, p_0, 'Without IRS')
        results[1].add_result(self.SNR_dB, r_irs, 'With IRS')
        results[1].add_result(self.SNR_dB, r_0, 'Without IRS')
        return results

    def optimize(self, SNR):
        phi = self.get_phi()
        vopt = (1 / np.sqrt(2)) * (np.ones((self.irs.antnum, 1), dtype=complex) +
                                   1j * np.ones((self.irs.antnum, 1), dtype=complex))
        objval_prev = math.inf
        objval = 0
        H = np.zeros((self.num_users(), self.bs.antnum), dtype=complex)
        Wopt = np.zeros((self.bs.antnum, self.num_users()), dtype=complex)
        iter = 1
        objval_max = 0
        vopt_max = vopt
        a = {}
        b = {}
        while iter < self.maxiter:
            for k in range(self.num_users()):
                H[:, k] = np.matmul(vopt.T, phi[k].conj()) + \
                          self.bs_user_links[k].get_channel_matrix()
            Wopt = self.get_mmse(H, SNR)
            for k in range(self.num_users()):
                for j in range(self.num_users()):
                    b[(j, k)] = np.matmul(self.bs_user_links[k].get_channel_matrix().conj(),
                                          Wopt[:, j])
                    a[(j, k)] = np.matmul(phi[k], Wopt[:, j])
            vopt_real = complex_to_real(vopt)
            res = optimize.minimize(get_cost_real,
                                    vopt_real,
                                    (a, b, self.num_users(), self.sigma_sqr, self.gamma),
                                    'Newton-CG',
                                    get_gradient_real)
            vopt = real_to_complex(res.x)
            vopt = np.exp(-1j*np.angle(vopt))
            objval_prev = objval
            objval = -get_cost(vopt, a, b, self.num_users(), self.sigma_sqr, 0)
            if objval > objval_max:
                objval_max = objval
                vopt_max = vopt
            error = np.abs(objval - objval_prev)
            print('Current error: ', error, ', objval: ', objval)
            if error < self.tol:
                break
            iter = iter + 1
        return vopt_max, Wopt

    def get_mmse(self, H, SNR):
        g = np.zeros((self.num_users(), 1), dtype=complex)
        H1 = np.zeros(H.shape, dtype=complex)
        for k in range(self.num_users()):
            g[k] = np.linalg.norm(H[:, k])/np.sqrt(self.sigma_sqr)
            H1[:, k] = H[:, k] / g[k]
        G = np.diag(g.T[0])
        Wopt = np.matmul(H1.conj().T,
                         np.matmul(np.linalg.inv(np.matmul(H1.conj().T, H1) +
                                                 (self.num_users()/SNR) * np.eye(self.bs.antnum)), G))
        norm = np.sqrt(SNR)/np.linalg.norm(Wopt, 'fro')
        return norm * Wopt

    def get_phi(self):
        if self.phi is None:
            phi = {}
            for k in range(self.num_users()):
                phi[k] = np.matmul(np.diag(self.irs_user_links[k].get_channel_matrix()[0].conj()),
                                   self.bs_irs_link.get_channel_matrix())
            self.phi = phi
        return self.phi

    def get_sum_rate(self, Wopt, vopt=None):
        sum_rate = 0
        if vopt is None:
            vopt = np.zeros((self.irs.antnum, 1))
        phi = self.get_phi()
        for k in range(self.num_users()):
            psig = np.power(np.abs(np.matmul(np.matmul(vopt.conj().T, phi[k]) +
                                             self.bs_user_links[k].get_channel_matrix().conj(), Wopt[:, k])), 2)
            pint = 0
            for j in range(self.num_users()):
                pint = pint + np.power(np.abs(np.matmul(np.matmul(vopt.conj().T, phi[k]) +
                                                        self.bs_user_links[k].get_channel_matrix().conj(),
                                                        Wopt[:, j])), 2)
            sum_rate = sum_rate + np.log2(1 + psig / (pint + self.sigma_sqr))
        return sum_rate


class Result:

    def __init__(self, x, y, text):
        self.x = x
        self.y = y
        self.text = text


class SimulationResult:

    def __init__(self, title,  xlabel, ylabel):
        self.results = []
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

    def add_result(self, x, y, text):
        self.results.append(Result(x, y, text))



