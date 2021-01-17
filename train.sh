#!/bin/bash
#SBATCH -N 1 # number of nodes
#SBATCH --gres=gpu:1 # number of GPUs to be allocated
#SBATCH -t 1-00:00 # time after which the process will be killed (D-HH:MM)
#SBATCH -o "/nfs/homedirs/%u/git/robust-gnns-at-scale/seml/train/output/slurm-%j.out"
#SBATCH --partition=gpu_all
#SBATCH --mem=256G 
#SBATCH --cpus-per-task=12
#SBATCH --qos=studentprio


cd ${SLURM_SUBMIT_DIR}
echo Starting job ${SLURM_JOBID}
echo SLURM assigned me these nodes:
squeue -j ${SLURM_JOBID} -O nodelist | tail -n +2
# Activate your conda environment if necessary
# conda init bash
# conda activate robustgnn
python script_train.py --config-files=seml/train/robustpprgo_local_test.yaml