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


W_TILDES="0.0025 0.0225 0.0425 0.0625 0.0825 0.1025 0.1225 0.1425 0.1625 0.1825 0.2025 0.2225 0.2425 0.2625 0.2825 0.3025 0.3225 0.3425 0.3625 0.3825 0.4025 0.4225 0.4425 0.4625 0.4825 0.5025 0.5225 0.5425 0.5625 0.5825 0.6025 0.6225 0.6425 0.6625 0.6825 0.7025 0.7225 0.7425 0.7625 0.7825 0.8025 0.8225 0.8425 0.8625 0.8825 0.9025 0.9225 0.9425 0.9625 0.9825 1.0025 1.0225 1.0425 1.0625 1.0825 1.1025 1.1225 1.1425 1.1625 1.1825 1.2025 1.2225 1.2425 1.2625 1.2825 1.3025 1.3225 1.3425 1.3625 1.3825 1.4025 1.4225 1.4425 1.4625 1.4825 1.5025 1.5225 1.5425 1.5625 1.5825 1.6025 1.6225 1.6425 1.6625 1.6825 1.7025 1.7225 1.7425 1.7625 1.7825 1.8025 1.8225 1.8425 1.8625 1.8825 1.9025 1.9225 1.9425 1.9625 1.9825 2.000"
NUM_CA3_NEURONS="50"

running_jobs_count() {
  ps r | wc -l
}
MAX_PARALLEL_JOBS=24

for NC3N in \$NUM_CA3_NEURONS; do
    for WT in \$W_TILDES; do
        echo "Running simulation with w_tilde = \${WT} num_ca3_neurons = \${NC3N}"
        
        python3 CA1network.py --w_tilde \${WT} --num_ca3_neurons \${NC3N} &
        
        while (( \$(running_jobs_count) >= MAX_PARALLEL_JOBS )); do
          sleep 1
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
cp -r ./ $DIR/run
cd $DIR/run

chmod 777 job.sh
nohup ./job.sh > ../job_output/log.txt

cd -
