#! /bin/bash

#PJM --rsc-list "node=1"
#PJM --mpi "max-proc-per-node=48"
#PJM -g hp200139
#PJM --rsc-list "elapse=00:40:00"
#PJM -m b,e
#PJM --mail-list "fukami.satoshi760@mail.kyutech.jp"
#PJM -j
#PJM -s 
#PJM -x PJM_LLIO_GFSCACHE=/vol0002

#module load Python3-CN

#. /vol0004/apps/oss/spack/share/spack/setup-env.sh
#
#module load Python3-CN
#export FLIB_CNTL_BARRIER_ERR=FALSE

#spack load /fhakchp # python@3.8.12%fj@4.7.0
#spack load /dgmiy5n # py-numpy@1.25.2%fj@4.10.0
#spack load /qqrwvm6 # py-scipy@1.8.1

#spack load /hcqvcsc # py-scikit-learn@1.3.2
#spack load /2h4rydm # py-matplotlib@3.3.4
#spack find --loaded # see the list of loaded modules
#
#export LD_LIBRARY_PATH=/lib64:

cd ../model
python3 CA1network.py

