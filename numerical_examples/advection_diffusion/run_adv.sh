#!/bin/bash

numTfinals=3
numKappas=5
numNoises=1
numNumBatches=3

python adv_deflation.py 0 0 0 0 'init'

for ((i = 0; i < $numNoises; i++)); do
	for ((j = 0; j < $numKappas; j++)); do
		for ((k = 0; k < $numTfinals; k++)) do
			python adv_deflation.py ${i} ${j} ${k} 0 'first'
			for ((l = 0; l < $numNumBatches; l++)) do
			  python adv_deflation.py ${i} ${j} ${k} ${l} 'psf'
			python adv_deflation.py ${i} ${j} ${k} 0 'last'
		done
	done
done