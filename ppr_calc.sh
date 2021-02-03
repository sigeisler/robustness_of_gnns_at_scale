#!/bin/bash
#SBATCH -N 1 # number of nodes
#SBATCH -t 0-02:00 # time after which the process will be killed (D-HH:MM)
#SBATCH -o "/nfs/homedirs/%u/git/robust-gnns-at-scale/seml/train/output/slurm-%j.out"
#SBATCH --mem=160G 
#SBATCH --cpus-per-task=20
#SBATCH --qos=interactive


cd ${SLURM_SUBMIT_DIR}
echo Starting job ${SLURM_JOBID}
echo SLURM assigned me these nodes:
squeue -j ${SLURM_JOBID} -O nodelist | tail -n +2
# Activate your conda environment if necessary
# conda init bash
# conda activate robustgnn
python pre_calc_ppr.py