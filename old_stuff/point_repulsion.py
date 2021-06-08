
import numpy as np
import matplotlib.pyplot as plt
from gaussblur import periodic_displacement, periodize_pts

d=2
num_pts = 200
gamma = 0.5


def phi(s):
    return np.power(s, -gamma)

def phi_prime(s):
    return -gamma * np.power(s, -(gamma+1.))

def force_profile(s):
    # return 0.05*np.power(s, -1.)
    return np.exp(-0.5*s)

def get_forces(xx,CC):
    ff = np.zeros(xx.shape)
    for i in range(xx.shape[1]):
        x = xx[:,i]
        dd = periodic_displacement(x, xx)
        ss = np.sum(dd * np.linalg.solve(CC[i,:,:], dd), axis=0)
        ss[i] = 1.
        ffi = force_profile(ss)
        ffi[i] = 0.
        ff = ff + ffi * dd
    return ff


xx = 2. * np.random.rand(d,num_pts) - 1

def sigma_fct(qq):
    return 2.*(0.07 + 0.04 * np.sin(np.pi * qq[0, :]) * np.cos(np.pi * qq[1, :]))

CC = np.einsum('i,jk->ijk', sigma_fct(xx), np.eye(d))

plt.figure()
plt.plot(xx[0,:], xx[1,:], '.')
CC = np.einsum('i,jk->ijk', sigma_fct(xx), np.eye(d))
ff = get_forces(xx,CC)
for i in range(num_pts):
    x = xx[:,i]
    f = ff[:,i]
    plt.arrow(x[0], x[1], f[0], f[1])

for _ in range(100):
    CC = np.einsum('i,jk->ijk', sigma_fct(xx), np.eye(d))
    ff = get_forces(xx,CC)
    xx = periodize_pts(xx + 0.01*ff)

###############################################################

# class PointPotential:
#     def __init__(me, xx, CC):
#         me.xx = xx
#         me.CC = CC
#
#         me.V = 0.
#         me.GG = np.zeros(me.xx.shape)
#         me.recompute()
#
#     def update_xx(me, new_xx):
#         me.xx = new_xx.copy()
#         me.recompute()
#
#     def recompute(me):
#         me.V = 0.
#         me.GG = np.zeros(me.xx.shape)
#         for i in range(xx.shape[1]):
#             x = me.xx[:, i]
#             zz = periodic_displacement(x, me.xx)  # displacements
#             ff = np.linalg.solve(me.CC[i,:,:], zz)
#             ss = np.sum(zz * ff, axis=0)
#             ss[i] = 1.
#
#             phis = phi(ss)
#             phis[i] = 0.
#             me.V = me.V + np.sum(phis)
#
#             dss = 2. * ff
#             # dss = 2. * zz
#             phi_primes = phi_prime(ss)
#             phi_primes[i] = 0.
#             # me.GG[:,i] = np.sum(phi_primes * dss, axis=1)
#             me.GG = me.GG + phi_primes * dss


# PP = PointPotential(xx,CC)

# plt.figure()
# plt.plot(PP.xx[0,:], PP.xx[1,:], '.')
# for i in range(num_pts):
#     x = PP.xx[:,i]
#     g = -0.002*PP.GG[:,i]
#     plt.arrow(x[0], x[1], g[0], g[1])

# for _ in range(100):
#     new_xx = periodize_pts(PP.xx + 0.0001 * PP.GG)
#     PP.update_xx(new_xx)
#
# V0 = np.copy(PP.V)
# GG0 = np.copy(PP.GG)
# dxx = np.random.randn(*xx.shape)
# s = 1e-8
# PP.update_xx(PP.xx + s*dxx)
# V1 = np.copy(PP.V)
# dV_diff = (V1-V0)/s
# dV = 2*np.sum((GG0 * dxx).reshape(-1))
# err_GG = np.abs(dV_diff - dV)/np.abs(dV_diff)
# print('s=',s, ', err_GG=', err_GG)
