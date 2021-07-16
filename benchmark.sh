#!/bin/bash -l
#SBATCH -D ./

### OUTPUTS, ERRORS AND JOB NAME:
#SBATCH -o /ptmp/pierocor/tmp_train_bench/bench_1_4_40.%j
#SBATCH -e /ptmp/pierocor/tmp_train_bench/bench_1_4_40.%j_err
#SBATCH -J bench_1_4_40

### TIME LIMIT:
#SBATCH --time=0-00:20:00

### NODE features:
### No need to modify them on raven!
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=0
#SBATCH --nodes=1
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


DATA_ROOT="/ptmp/pierocor/datasets"
WEIGHTS_ROOT="/ptmp/pierocor/tmp_train_bench/weights"
LOGS_ROOT="/ptmp/pierocor/tmp_train_bench/logs"
SAMPLE_ROOT="/ptmp/pierocor/tmp_train_bench/samples"

### Run the program:
srun python benchmark.py \
  --data_root $DATA_ROOT \
  --weights_root $WEIGHTS_ROOT \
  --logs_root $LOGS_ROOT \
  --samples_root $SAMPLE_ROOT \
  --num_epochs 5 \
  --dataset ecoset_cs500 \
  --shuffle  --batch_size 40 \
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
  --seed 0 --copy_in_mem  --num_workers 0

