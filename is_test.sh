#!/bin/bash -l
#SBATCH -D ./
## #SBATCH --array=0-5%1

### OUTPUTS, ERRORS AND JOB NAME:
### Replace these with a path to an existing folder where you have writing access.
### The %j will be replaced by a unique SLURM job id (useful to avoid ovewriting existing files)
#SBATCH -o /ptmp/wero/eval_fix/output/is_test.%j ##_%A_%a
#SBATCH -e /ptmp/wero/eval_fix/output/is_test.%j ##_%A_%a
#SBATCH -J is_test

### TIME LIMIT: e.g.
### 1-00:00:00 -> 1 day (Maximum)
### 0-00:20:00 -> 20 minutes
#SBATCH --time=00-00:20:00
#SBATCH --signal=USR1@300

#SBATCH --mail-type=all
#SBATCH --mail-user=wklos@uos.de

### NODE features:
### Num nodes, num tasks per node
#SBATCH --nodes=15
#SBATCH --ntasks-per-node=4

### No need to modify below on raven!
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=0
#SBATCH --ntasks-per-socket=2
#SBATCH --cpus-per-task=18
#SBATCH --threads-per-core=1

### Modules and env variables
source raven.env

### print loaded modules and basic SLURM info
module list

echo -e "Nodes: ${SLURM_JOB_NUM_NODES} \t NTASK: ${SLURM_NTASKS}"
echo "${SLURM_NODELIST}"


DATA_ROOT="/ptmp/pierocor/datasets"  # This should work but you can use a different one
WEIGHTS_ROOT="/ptmp/wero/eval_fix/weights"  # Replace by a folder where you have writing access (HVD folder)
LOGS_ROOT="/ptmp/wero/eval_fix/logs"  # Replace by a folder where you have writing access (HVD folder)
SAMPLE_ROOT="/ptmp/wero/eval_fix/samples"  # Replace by a folder where you have writing access (HVD folder)
# Replace by the backup folder of your best run
LOAD_FROM="/ptmp/wero/eval_fix/weights/BigGAN_ecoset_cs500__mem_w0_seed12_Gch96_Dch96_bs1760_hvd40_4_nDs2_Glr5.0e-05_Dlr3.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitortho_Dinitortho_Gattn64_Dattn64_Gshared_hier_ema/"

### Run the program:
### Change wathever you want and have fun!
srun python test.py \
  --data_root $DATA_ROOT \
  --weights_root $WEIGHTS_ROOT \
  --logs_root $LOGS_ROOT \
  --samples_root $SAMPLE_ROOT \
  --num_epochs 900 \
  --dataset ecoset_cs500 \
  --shuffle  --num_workers 0 --batch_size 1760 \
  --num_G_accumulations 1 --num_D_accumulations 1 \
  --num_D_steps 2 --G_lr 5e-5 --D_lr 3e-4 --D_B2 0.999 --G_B2 0.999 \
  --G_attn 64 --D_attn 64 \
  --G_nl inplace_relu --D_nl inplace_relu \
  --SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
  --G_ortho 0.0 \
  --G_shared \
  --G_init ortho --D_init ortho \
  --hier --dim_z 140 --shared_dim 128 \
  --G_eval_mode \
  --G_ch 96 --D_ch 96 \
  --ema --use_ema --ema_start 20000 \
  --test_every 1 --save_every 1000 \
  --num_best_copies 5 --num_save_copies 2 \
  --copy_in_mem \
  --seed 12 --experiment_name is_test --resume \
  --load_from ${LOAD_FROM} --load_weights best0 --num_inception_images 200000

### DO NOT USE PARALLEL!!!

### RESUME:
## if you run the same config twice and you use the --resume flag, it will
## load the last checkpoint. Optionally you can load weights from a previous
## run without overwriting its logs and weights by using --load_from. The new
## run will have its own logs and weights folder as specified by "experiment_name".
##
## E.g. substitute last line with:
## --seed 0 --resume \
## --load_from /ptmp/pierocor/BigGan_out/weights/BigGAN_ecoset_cs500_seed40_Gch96_Dch96_bs176_nDs2_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitortho_Dinitortho_Gattn64_Dattn64_Gshared_hier_ema_hinge


