#!/bin/bash -l
#SBATCH -D ./
#SBATCH -o /u/pierocor/work/BigGAN-PyTorch/tmp/output/array_1_E256_176_p.%A_%a
#SBATCH -e /u/pierocor/work/BigGAN-PyTorch/tmp/output/array_1_E256_176_p.%A_%a
#SBATCH -J arr_1_E256_176_w8_p
#SBATCH --time=0:10:00
#SBATCH --array=0-2%1

# Node feature:
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --ntasks-per-socket=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --threads-per-core=1

### Debug and Analytics
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1

### Modules and env variables
source /u/pierocor/work/BigGAN-PyTorch/raven.env

module list

echo -e "Nodes: ${SLURM_JOB_NUM_NODES} \t NTASK: ${SLURM_NTASKS}"
echo "${SLURM_NODELIST}"

RESUME=""
if [ $SLURM_ARRAY_TASK_ID -ne 0 ]; then
  RESUME="--resume"
fi

# Run the program:
srun python train.py \
  --data_root /ptmp/pierocor/datasets/\
  --num_epochs 1 \
  --dataset E256_hdf5 \
  --shuffle  --num_workers 8 --batch_size 176 \
  --num_G_accumulations 1 --num_D_accumulations 1 \
  --num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
  --G_attn 64 --D_attn 64 \
  --G_nl inplace_relu --D_nl inplace_relu \
  --SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
  --G_ortho 0.0 \
  --G_shared \
  --G_init ortho --D_init ortho \
  --hier --dim_z 120 --shared_dim 128 \
  --G_eval_mode \
  --G_ch 96 --D_ch 96 \
  --ema --use_ema --ema_start 300 \
  --test_every 120 --save_every 120 --num_best_copies 5 --num_save_copies 2 --seed 0 \
  --use_multiepoch_sampler  --parallel ${RESUME}

