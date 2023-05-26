#!/bin/bash

#ll, ii, jj, kk
#nb, ns, kp, tf
#failures:
#[[0 0 0 0]
# [0 0 1 1]
# [1 0 0 2]
# [1 0 3 0]
# [2 0 4 1]]

#python adv_deflation.py 0 0 0 0 'first'
#python adv_deflation.py 0 0 0 0 'psf'
#python adv_deflation.py 0 0 0 0 'last'

#python adv_deflation.py 0 1 1 0 'first'
#python adv_deflation.py 0 1 1 0 'psf'
#python adv_deflation.py 0 1 1 0 'last'

#python adv_deflation.py 0 0 2 0 'first'
#python adv_deflation.py 0 0 2 1 'psf'
#python adv_deflation.py 0 0 2 0 'last'

#python adv_deflation.py 0 3 0 0 'first'
#python adv_deflation.py 0 3 0 1 'psf'
#python adv_deflation.py 0 3 0 0 'last'

python adv_deflation.py 0 4 1 0 'first'
python adv_deflation.py 0 4 1 2 'psf'
python adv_deflation.py 0 4 1 0 'last'
