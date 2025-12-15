#!/bin/bash
#SBATCH --job-name="FisH-combo"
#SBATCH --chdir=/home/lucentlab/tsavitski/FisH
#SBATCH --nodes=1
#SBATCH --time=7-00:40:00
#SBATCH --output=/home/lucentlab/tsavitski/FisH/sim.log
#SBATCH --partition=gpu

echo "Date              = $(date)"
echo "Working Directory = $(pwd)"
echo "Node              = $(hostname)"
echo ""
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"
echo "Slurm Job Id                   = $SLURM_JOB_ID"

source /home/lucentlab/tsavitski/.bashrc
conda activate redacules
python simulate_systems.py combo 8