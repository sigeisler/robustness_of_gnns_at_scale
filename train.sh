#!/bin/bash
#SBATCH -N 1 # number of nodes
#SBATCH --gres=gpu:1 # number of GPUs to be allocated
#SBATCH -t 0-02:00 # time after which the process will be killed (D-HH:MM)
#SBATCH -o "/nfs/homedirs/%u/slurm-output/slurm-%j.out"
#SBATCH --partition=gpu_all
#SBATCH --mem=120000 # the memory (MB) that is allocated to the job. If your job exceeds this it will be killed
#        -- but don't set it too large since it will block resources and will lead to your job being given a low
#           priority by the scheduler.

cd ${SLURM_SUBMIT_DIR}
echo Starting job ${SLURM_JOBID}
echo SLURM assigned me these nodes:
squeue -j ${SLURM_JOBID} -O nodelist | tail -n +2
# Activate your conda environment if necessary
# conda init bash
# conda activate robustgnn
python script_train.py --config-files=seml/train/robustpprgo_local_test.yaml