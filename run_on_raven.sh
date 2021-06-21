#!/bin/bash -l
#SBATCH -D ./

### OUTPUTS, ERRORS AND JOB NAME:
### Replace these with a path to an existing folder where you have writing access.
### The %j will be replaced by a unique SLURM job id (useful to avoid ovewriting existing files)
#SBATCH -o hvd_BigGan.%j
#SBATCH -e hvd_BigGan.%j
#SBATCH -J hvd_BigGan

### TIME LIMIT: e.g.
### 1-00:00:00 -> 1 day (Maximum)
### 0-00:20:00 -> 20 minutes
#SBATCH --time=1-00:00:00

### NODE features:
### No need to modify them on raven!
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=0
#SBATCH --nodes=4
#SBATCH --ntasks-per-socket=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --threads-per-core=1

### Modules and env variables
source raven.env

### print loaded modules and basic SLURM info
module list

echo -e "Nodes: ${SLURM_JOB_NUM_NODES} \t NTASK: ${SLURM_NTASKS}"
echo "${SLURM_NODELIST}"


DATA_ROOT="/ptmp/pierocor/datasets"  # This should work but you can use a different one
WEIGHTS_ROOT="/ptmp/pierocor/BigGan_out/hvd/weights"  # Replace by a folder where you have writing aaccess
LOGS_ROOT="/ptmp/pierocor/BigGan_out/hvd/logs"  # Replace by a folder where you have writing aaccess
SAMPLE_ROOT="/ptmp/pierocor/BigGan_out/hvd/samples"  # Replace by a folder where you have writing aaccess

### Run the program:
### Change wathever you want and have fun!
srun python train.py \
  --data_root $DATA_ROOT \
  --weights_root $WEIGHTS_ROOT \
  --logs_root $LOGS_ROOT \
  --samples_root $SAMPLE_ROOT \
  --num_epochs 100 \
  --dataset E256_hdf5 \
  --shuffle  --num_workers 8 --batch_size 704 \
  --num_G_accumulations 1 --num_D_accumulations 1 \
  --num_D_steps 2 --G_lr 4e-4 --D_lr 1.6e-3 --D_B2 0.999 --G_B2 0.999 \
  --G_attn 64 --D_attn 64 \
  --G_nl inplace_relu --D_nl inplace_relu \
  --SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
  --G_ortho 0.0 \
  --G_shared \
  --G_init ortho --D_init ortho \
  --hier --dim_z 120 --shared_dim 128 \
  --G_eval_mode \
  --G_ch 96 --D_ch 96 \
  --ema --use_ema --ema_start 20000 \
  --test_every 1000 --save_every 1000 \
  --num_best_copies 5 --num_save_copies 2 \
  --seed 0
### DO NOT USE PARALLEL!!!

### RESUME:
### if you run the same config twice and you use the --resume flag, it will load the last checkpoint

