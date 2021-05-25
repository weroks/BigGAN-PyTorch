#!/bin/bash -l
#SBATCH -D ./
#SBATCH -o /u/pierocor/work/BigGAN-PyTorch/tmp/output/E256.%j
#SBATCH -e /u/pierocor/work/BigGAN-PyTorch/tmp/output/E256.%j
#SBATCH -J E256
#SBATCH --time=00:10:00
#SBATCH -p gpudev
# Node feature:
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-socket=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --threads-per-core=1

### Debug and Analytics
export HPCMD_SLEEP=0
export HPCMD_AWAKE=60
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1

### Modules and env variables
source .env

module list

echo -e "Nodes: ${SLURM_JOB_NUM_NODES} \t NTASK: ${SLURM_NTASKS}"
echo "${SLURM_NODELIST}"

# echo "Copying dataset to /tmp ..."
# cp /ptmp/pierocor/datasets/small_E256.hdf5 /tmp/
# echo "Done!"

# Run the program:

	# --dataset E256_hdf5 \
	# --dataset small_E256_hdf5 --load_in_mem\

srun hpcmd_slurm python train.py \
    --data_root /ptmp/pierocor/datasets/ \
	--num_epochs 1 \
	--dataset E256_hdf5 \
	--shuffle  --num_workers 8 --batch_size 34 \
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
	--test_every 2000 --save_every 1000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
	--use_multiepoch_sampler

# echo "Removing dataset to /tmp ..."
# rm /tmp/small_E256.hdf5
# echo "Done!"
