import numpy as np
import matplotlib.pyplot as plt

tt = np.linspace(-1, 2, 1000)
ff1 = tt/(1. + (2.*tt - 1.)**(2**1))
ff2 = tt/(1. + (2.*tt - 1.)**(2**2))
ff3 = tt/(1. + (2.*tt - 1.)**(2**3))
ff4 = tt/(1. + (2.*tt - 1.)**(2**4))
ff5 = tt/(1. + (2.*tt - 1.)**(2**5))
ff6 = tt/(1. + (2.*tt - 1.)**(2**6))
ff7 = tt/(1. + (2.*tt - 1.)**(2**7))

plt.figure(figsize=(8,4))
plt.plot(tt, ff3, '-.k')
plt.plot(tt, ff5,'--k')
plt.plot(tt, ff7, 'k')
plt.legend(['k=3', 'k=5', 'k=7'])
plt.xlabel(r'$\lambda$')
plt.ylabel(r'$\widetilde{f}(\lambda)$')
plt.title(r'Rational function $\widetilde{f}(\lambda)$')

plt.savefig('spd_rational_function.pdf', bbox_inches='tight', dpi=100)

np.savetxt('spd_rational_function_tt.txt', tt)
np.savetxt('spd_rational_function_ff3.txt', ff3)
np.savetxt('spd_rational_function_ff5.txt', ff5)
np.savetxt('spd_rational_function_ff7.txt', ff7)