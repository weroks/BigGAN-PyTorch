#!/bin/bash -l
#SBATCH -D ./
#SBATCH -o /ptmp/pierocor/BigGan_out//output//E256_174_full_w8_p.%A_%a
#SBATCH -e /ptmp/pierocor/BigGan_out//output//E256_174_full_w8_p.%A_%a
#SBATCH -J E256_174_full_w8_p
#SBATCH --time=1-00:00:00

#SBATCH --array=0-5%1
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


### store job submit script using a unique id:
if [ -z "$SLURM_ARRAY_JOB_ID" ]; then
  cp /ptmp/pierocor/BigGan_out//subscripts//run.sh /ptmp/pierocor/BigGan_out//subscripts//E256_174_full_w8_p.${SLURM_JOB_ID}.sh
elif [[ ${SLURM_ARRAY_TASK_ID} -eq 0 ]]; then
  cp /ptmp/pierocor/BigGan_out//subscripts//run.sh /ptmp/pierocor/BigGan_out//subscripts//E256_174_full_w8_p.${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.sh
fi

### print modules and basic SLURM info
module list

echo -e "Nodes: ${SLURM_JOB_NUM_NODES} \t NTASK: ${SLURM_NTASKS}"
echo "${SLURM_NODELIST}"

### RESUME variable defined for job arrays only, if not empty

if [ $SLURM_ARRAY_TASK_ID -ne 0 ]; then
  RESUME="--resume"
fi


### Run the program:
srun python train.py \
  --data_root /ptmp/pierocor/datasets/ \
  --weights_root /ptmp/pierocor/BigGan_out//weights/ \
  --logs_root /ptmp/pierocor/BigGan_out//logs/ \
  --samples_root /ptmp/pierocor/BigGan_out//samples/ \
  --num_epochs 100 \
  --dataset E256_hdf5 \
  --shuffle  --num_workers 8 --batch_size 174 \
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
  --ema --use_ema --ema_start 20000 \
  --test_every 1000 --save_every 1000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
  --use_multiepoch_sampler  --parallel ${RESUME}

