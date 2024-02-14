import numpy as np

Ls = ['1.0', '0.25', '0.1111111111111111']

all_all_fro_errs = [np.loadtxt('all_fro_errors_L='+Ls[i]+'.txt') for i in range(len(Ls))]
all_all_HODLR_costs = [np.loadtxt('all_HODLR_costs_L='+Ls[i]+'.txt') for i in range(len(Ls))]

all_err_levels = np.loadtxt('all_error_levels.txt')

for i in range(len(Ls)):
    print("L = "+Ls[i]+' (length scale)')
    for j in range(len(all_err_levels)):
        idx = np.argmin(abs(all_all_fro_errs[i]-all_err_levels[j]))
        print("error level = {0:1.2e}".format(all_err_levels[j]))
        print("HODLR cost = {0:1.1e}".format(all_all_HODLR_costs[i][idx]))
        print("")
        #print("actual error = {0:1.2e}".format(all_all_fro_errs[i][idx]))


