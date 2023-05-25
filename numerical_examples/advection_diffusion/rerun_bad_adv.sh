#!/bin/bash

#ll, ii, jj, kk
#nb, ns, kp, tf
#failures:
#[[0 0 3 2]
# [0 0 4 0]
# [1 0 3 2]
# [1 0 4 2]
# [2 0 0 1]]

#python adv_deflation.py 0 0 1 0 'first'
#python adv_deflation.py 0 0 1 2 'psf'
#python adv_deflation.py 0 0 1 0 'last'

#python adv_deflation.py 0 4 0 0 'first'
#python adv_deflation.py 0 4 0 0 'psf'
#python adv_deflation.py 0 4 0 0 'last'

#python adv_deflation.py 0 3 2 0 'first'
#python adv_deflation.py 0 3 2 0 'psf'
#python adv_deflation.py 0 3 2 1 'psf'
#python adv_deflation.py 0 3 2 0 'last'

python adv_deflation.py 0 4 2 0 'first'
python adv_deflation.py 0 4 2 1 'psf'
python adv_deflation.py 0 4 2 0 'last'
