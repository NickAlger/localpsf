import numpy as np

exts = ["none", "reg", "psf1", "psf5", "psf25"]
ndata = len(exts)
datas = [np.loadtxt("ee_"+exts[i]+"_0.049999999999999996") for i in range(ndata)]
N = len(datas[0])


f = open("generalizedEigenvalues_0.05noise.dat", "w")
f.write("k ")
for i in range(ndata):
    f.write(exts[i] + " ")
f.write("\n")

for i in range(N):
    f.write("{0:d} {1:f} {2:f} {3:f} {4:f} {5:f}\n".format(i, datas[0][i], datas[1][i], datas[2][i], datas[3][i], datas[4][i]))





