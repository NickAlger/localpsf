import numpy as np
from localpsf_helpers import points_which_are_not_in_ellipsoid_numba

run_test = False

class BatchProductConvolution:
    def __init__(me, eval_dirac_comb_responses_eta, eval_weighting_functions_w,
                 sample_point_batches_xx, mean_batches_mu, covariance_batches_Sigma, num_standard_deviations_tau):
        me.eval_eta_batches = eval_dirac_comb_responses_eta
        me.eval_ww_batches = eval_weighting_functions_w
        me.point_batches = sample_point_batches_xx
        me.mean_batches = mean_batches_mu
        me.covariance_batches = covariance_batches_Sigma
        me.tau = num_standard_deviations_tau

    @property
    def num_batches(me):
        return len(me.eval_eta_batches)

    def compute_product_convolution_entries(me, yy, xx):
        # Evaluates H(yk,xk) for many point pairs (yk,xk)
        hh = np.zeros(xx.shape[0])
        for b in range(me.num_batches):
            hh = hh + me._compute_product_convolution_entries_one_batch(yy, xx, b)
        return hh

    def _compute_product_convolution_entries_one_batch_slow(me, yy, xx, b):
        # for testing purposes
        eval_eta = me.eval_eta_batches[b]
        eval_ww = me.eval_ww_batches[b]
        pp = me.point_batches[b]
        mu = me.mean_batches[b]
        Sigma = me.covariance_batches[b]
        num_batch_points = len(eval_ww)
        num_eval_points = xx.shape[0]

        W = np.zeros((num_batch_points, num_eval_points))
        Psi = np.zeros((num_batch_points, num_eval_points))
        for ii in range(num_batch_points):
            for kk in range(num_eval_points):
                z = pp[ii,:] + yy[kk,:] - xx[kk,:]
                if np.dot(z - mu[ii,:], np.linalg.solve(Sigma[ii,:,:], z - mu[ii,:])) < me.tau**2:
                    Psi[ii,kk] = eval_eta(z.reshape((1,-1)))
                W[ii,kk] = eval_ww[ii](xx[kk,:].reshape((1,-1)))

        hh = np.zeros(num_eval_points)
        for ii in range(num_batch_points):
            hh = hh + Psi[ii,:]*W[ii,:]
        return hh

    def _compute_product_convolution_entries_one_batch(me, yy, xx, b):
        eval_eta = me.eval_eta_batches[b]
        eval_ww = me.eval_ww_batches[b]
        pp = me.point_batches[b]
        mu = me.mean_batches[b]
        Sigma = me.covariance_batches[b]
        num_batch_points = len(eval_ww)
        num_eval_points = xx.shape[0]

        ww_at_xx = []
        for eval_w in eval_ww:
            ww_at_xx.append(eval_w(xx))

        all_nonzero_zz = []
        all_nonzero_k_inds = []
        for ii in range(num_batch_points):
            zz = pp[ii,:].reshape((1,-1)) + yy - xx
            eval_inds_in_ellipsoid = np.logical_not(points_which_are_not_in_ellipsoid_numba(Sigma[ii,:,:], mu[ii,:], zz, me.tau))
            all_nonzero_zz.append(zz[eval_inds_in_ellipsoid,:])
            all_nonzero_k_inds.append(eval_inds_in_ellipsoid)

        nonzero_zz_vector = np.vstack(all_nonzero_zz)
        # print('len(nonzero_zz_vector)=', len(nonzero_zz_vector), ', num_batch_points*num_eval_points=', num_batch_points*num_eval_points)
        nonzero_Phi = _unpack_vector((z.shape[0] for z in all_nonzero_zz), eval_eta(nonzero_zz_vector))

        hh = np.zeros(num_eval_points)
        for nonzero_k_inds, w_at_xx, nonzero_phi_k in zip(all_nonzero_k_inds, ww_at_xx, nonzero_Phi):
            hh[nonzero_k_inds] = hh[nonzero_k_inds] + w_at_xx[nonzero_k_inds] * nonzero_phi_k
        return hh


def _unpack_vector(lengths, v):
    # Example:
    #   v = np.arange(3 + 2 + 4 + 5)
    #   vv = unpack_vector((3,2,4,5), v)
    #   print(vv)
    # -> [array([0., 1., 2.]), array([3., 4.]), array([5., 6., 7., 8.]), array([ 9., 10., 11., 12., 13.])]
    ww = []
    start = 0
    for l in lengths:
        stop = start + l
        ww.append(v[start:stop])
        start = stop
    return ww


if run_test:
    from time import time
    eval_eta1 = lambda x: np.sin(x[:,0])*np.cos(x[:,1])
    eval_eta2 = lambda x: x[:,0]**2 + 2.3*x[:,1]*x[:,0]
    eval_eta3 = lambda x: np.linalg.norm(x, axis=1)

    eval_dirac_comb_responses_eta = [eval_eta1, eval_eta2, eval_eta3]
    sample_point_batches_xx = [np.random.rand(5,2), np.random.rand(1,2),np.random.rand(2,2)]

    w1_1 = lambda x: np.sin(x[:,0] + x[:,1])
    w1_2 = lambda x: np.sin(x[:,0] - x[:,1])
    w1_3 = lambda x: (x[:,0] + 0.3*x[:,1])**2
    w1_4 = lambda x: x[:,0]
    w1_5 = lambda x: x[:,1]

    w2_1 = lambda x: 0.1*x[:,0] + x[:,1] - 4.

    w3_1 = lambda x: np.exp(x[:,0]) + x[:,1]
    w3_2 = lambda x: np.log(np.abs(1. + x[:,0]*x[:,1]))

    eval_weighting_functions_w = [[w1_1, w1_2, w1_3, w1_4, w1_5], [w2_1], [w3_1, w3_2]]

    mean_batches_mu = [np.random.rand(5,2), np.random.rand(1,2),np.random.rand(2,2)]

    def random_spd_matrix():
        U, ss, _ = np.linalg.svd(np.random.randn(2,2))
        return np.dot(U, np.dot(np.diag(ss), U.T))

    Sigma1 = np.zeros((5,2,2))
    for k in range(5):
        Sigma1[k,:,:] = random_spd_matrix()

    Sigma2 = np.zeros((1,2,2))
    for k in range(1):
        Sigma2[k,:,:] = random_spd_matrix()

    Sigma3 = np.zeros((2,2,2))
    for k in range(2):
        Sigma3[k,:,:] = random_spd_matrix()

    covariance_batches_Sigma = [Sigma1, Sigma2, Sigma3]

    num_standard_deviations_tau = 0.5

    PC = BatchProductConvolution(eval_dirac_comb_responses_eta, eval_weighting_functions_w,
                                 sample_point_batches_xx, mean_batches_mu, covariance_batches_Sigma, num_standard_deviations_tau)


    xx = np.random.rand(5000,2)
    yy = np.random.rand(5000,2)
    t = time()
    hha1 = PC._compute_product_convolution_entries_one_batch_slow(yy,xx,0)
    dt1 = time() - t
    print('dt1=', dt1)
    t = time()
    hha2 = PC._compute_product_convolution_entries_one_batch(yy,xx,0)
    dt2 = time() - t
    print('dt2=', dt2)
    err1 = np.linalg.norm(hha1 - hha2)
    print('err1=', err1)

    hhb1 = PC._compute_product_convolution_entries_one_batch_slow(yy,xx,1)
    hhb2 = PC._compute_product_convolution_entries_one_batch(yy,xx,1)
    err2 = np.linalg.norm(hhb1 - hhb2)
    print('err2=', err2)

    hhc1 = PC._compute_product_convolution_entries_one_batch_slow(yy,xx,2)
    hhc2 = PC._compute_product_convolution_entries_one_batch(yy,xx,2)
    err3 = np.linalg.norm(hhc1 - hhc2)
    print('err3=', err3)

    h = PC.compute_product_convolution_entries(yy, xx)
    h2 = hha1 + hhb1 + hhc1
    err_total = np.linalg.norm(h - h2)
    print('err_total=', err_total)