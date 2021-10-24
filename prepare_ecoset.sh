#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./small_eco.%j
#SBATCH -e ./small_eco.%j
# Initial working directory:
#SBATCH -D ./

# Job Name:
#SBATCH -J small_eco
#
# Number of nodes and MPI tasks per node:
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-socket=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#
#SBATCH --mail-type=all
#SBATCH --mail-user=weronik97@zedat.fu-berlin.de
#
# Wall clock limit (max. is 24 hours):
#SBATCH --time=01:00:00

# Load compiler and MPI modules (must be the same as used for compiling the code)
source raven.env


## python make_hdf5.py --dataset E256 --batch_size 256 --data_root data --num_workers 8
python calculate_inception_moments.py --dataset ecoset_cs500 --data_root /ptmp/pierocor/datasets
