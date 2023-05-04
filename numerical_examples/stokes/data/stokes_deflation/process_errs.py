import numpy as np



exts = ["none", "reg", "psf1", "psf5", "psf25"]

for ext in exts:
    data = np.loadtxt("errs_" + ext + "_0.049999999999999996") # 5% noise case
    # append initial relative residual of 1.0, make into ordered pairs and export the data 
    datapairs = np.zeros((len(data)+1, 2))
    datapairs[0, :] = [0, 1.0]
    for i in range(len(data)):
        datapairs[i+1, :] = [i+1, data[i]]
    np.savetxt("cg_relres_"+ext+"_0.05noise.dat", datapairs)
