import numpy as np
import matplotlib.pyplot as plt

save=True
all_kk=[1,2,3]
a = -1.0
b = 0.0

linestyles = ['solid', 'dashed', 'dotted', 'dashdot', 'loosely dotted', 'loosely dashed']

all_ff = []
tt = np.linspace(a - (b-a), b + (b-a), 1000)
for kk in all_kk:
    ff = 1. / (1. + ((2. * tt - (b + a))/(b - a)) ** (2 ** kk))
    all_ff.append(ff)


legend = []
plt.figure(figsize=(8,4))
for ii in range(len(all_kk)):
    kk = all_kk[ii]
    ff = all_ff[ii]
    plt.plot(tt, ff, c='k', linestyle=linestyles[ii])
    legend.append('k=' + str(kk))

# for kk, ff in zip(all_kk, all_ff):
#     plt.plot(tt, ff)
#     legend.append('k='+str(kk))
plt.legend(legend)
plt.xlabel(r'$\lambda$')
plt.ylabel(r'$\Pi_k(\lambda)$')
plt.title(r'Rational function $\Pi_k(\lambda)$')

if save:
    plt.savefig('spd_rational_function.pdf', bbox_inches='tight', dpi=100)
    np.savetxt('spd_rational_function_tt.txt', tt)
    for kk, ff in zip(all_kk, all_ff):
        np.savetxt('spd_rational_function_ff'+str(kk)+'.txt', ff)

####

# a = 0.5
# b = 1.0
# k=1
# N = 2**k
#
# cc = np.linalg.solve(np.array([[1., a],[1., b]]), np.array([1., np.power(2., 1./N)]))
#
# f = lambda t: 1./np.power(cc[0] + cc[1]*t, N)
#
# plt.plot(tt, f(tt))
# plt.ylim(-0.1, 1.1)