#!/bin/bash
#SBATCH -N 1 # number of nodes
#SBATCH -t 1-00:00 # time after which the process will be killed (D-HH:MM)
#SBATCH -o "/nfs/homedirs/%u/git/robust-gnns-at-scale/seml/train/output/slurm-%j.out"
#SBATCH --mem=380G 
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu
#SBATCH --qos=studentpriocpu


cd ${SLURM_SUBMIT_DIR}
echo Starting job ${SLURM_JOBID}
echo SLURM assigned me these nodes:
squeue -j ${SLURM_JOBID} -O nodelist | tail -n +2
# Activate your conda environment if necessary
# conda init bash
# conda activate robustgnn
python scripts/ppr_batched_ppr_to_storage.py