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


W_TILDES="1.6"
NUM_CA1_NEURONS="5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100"
SEEDS="0 1 2 3 4 5"
NUM_CA3_NEURONS="100"

running_jobs_count() {
  ps r | wc -l
}
MAX_PARALLEL_JOBS=24

for SEED in $SEEDS; do
  for NC3N in $NUM_CA3_NEURONS; do
    for WT in $W_TILDES; do
      for NC1N in $NUM_CA1_NEURONS; do
        echo "Running simulation with w_tilde = ${WT} num_ca3_neurons = ${NC3N} num_ca1_neurons = ${NC1N}"
        
        python3 CA1network.py --w_tilde ${WT} --num_ca3_neurons ${NC3N} --num_ca1_neurons ${NC1N} --seed ${SEED} &
        
        while (( $(running_jobs_count) >= MAX_PARALLEL_JOBS )); do
          sleep 1
        done
      done
    done
  done
done

wait

echo "全てのシミュレーションが完了しました。"

