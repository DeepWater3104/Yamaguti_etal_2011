cat << EOF > job.sh
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
#export LD_LIBRARY_PATH=/lib64:$LD_LIBRARY_PATH

cd ../model


W_TILDES="0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0"
T_INTERVAL_T="25 50 75 100 125 150 175 200"
SEEDS="0 1 2 3 4 5"


running_jobs_count() {
  ps r | wc -l
}
MAX_PARALLEL_JOBS=24

for WT in \$W_TILDES; do
  for INT in \$T_INTERVAL_T; do
    for SEED in \$SEEDS; do
      echo "Running simulation with w_tilde = \${WT} t_interval_T = \${INT} seed= \${SEED}"
      
      python3 CA1network.py --w_tilde \${WT} --t_interval_T \${INT} --seed \${SEED} &
      
      while (( \$(running_jobs_count) >= MAX_PARALLEL_JOBS )); do
        sleep 1
      done
    done
  done
done

wait

echo "全てのシミュレーションが完了しました。"

EOF

#TMP="/vol0206/data/hp200139/u12103/Yamaguti_etal_2011/record/results_"
TMP="/home/satoshi/Yamaguti_etal_2011/record/results_"
DATE=`date '+%Y-%m-%d-'`
TIME=`date '+%H-%M'`
DIR=$TMP$DATE$TIME
mkdir $DIR
mkdir $DIR/data
mkdir $DIR/figure
JOB_OUTPUT_DIR=$DIR/job_output
mkdir $JOB_OUTPUT_DIR
cp -r ../model $DIR/model
cp -r ../LDA $DIR/LDA
cp -r ./ $DIR/run
cd $DIR/run

chmod 777 job.sh
./job.sh

cd -
