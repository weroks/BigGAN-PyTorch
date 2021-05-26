#!/bin/bash -l

# Default values

DATASET='small_E256'
BS='34'
MODE='test'
NUM_WORKERS='8'
MONITORING="sys"

# PATHS
DATA_ROOT="/ptmp/pierocor/datasets/"

PROJ_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SUBSCRIPTS_DIR="${PROJ_ROOT}/tmp/subscripts/" 
OUTPUT_DIR="${PROJ_ROOT}/tmp/output/"

print_usage() {
  printf "Usage: ./submit.sh
 -m mode (test | long) (10 mins on gpudev vs 3h on gpu1_v100);\
 -d <string> dataset (E256 | small_E256);\
 -b <int> batch size;\
 -w <int> number of workers for DataLoader;\
 -t <string> Monitoring tool (sys | usr | pt)"
}

while getopts ':m:d:b:w:t:' flag; do
  case "${flag}" in
    m) MODE="${OPTARG}" ;;
    d) DATASET="${OPTARG}" ;;
    b) BS="${OPTARG}" ;;
    w) NUM_WORKERS="${OPTARG}" ;;
    t) MONITORING="${OPTARG}" ;;
    *) print_usage
       exit 1 ;;
  esac
done

if [[ ${MODE} == "long" ]]; then
  JOB_QUEUE=''
  JOB_TIME='#SBATCH --time=03:00:00'
else
  JOB_QUEUE='#SBATCH -p gpudev'
  JOB_TIME='#SBATCH --time=0:05:00'
  MODE='test'
fi

if [[ ${DATASET} == "E256" ]]; then
  DATA_ARG="--dataset E256_hdf5 \\"
elif [[ ${DATASET} == "small_E256" ]]; then
  DATA_ARG="--dataset small_E256_hdf5 --load_in_mem \\"
else
  echo "Wrong dataset name"
  exit 1
fi

case $MONITORING in
  sys) RUN="srun python" ;;
  usr) RUN="export HPCMD_SLEEP=0; export HPCMD_AWAKE=10; srun hpcmd_slurm python" ;;
  pt) RUN="srun hpcmd_suspend python -m torch.utils.bottleneck" ;;
  *) echo "Worng monitoring tool. Available: sys, usr, pt."
     exit 2 ;;
esac

JOB_NAME="${DATASET}_${BS}_${MODE}_w${NUM_WORKERS}"

mkdir -p ${SUBSCRIPTS_DIR}
mkdir -p ${OUTPUT_DIR}

cat > ${SUBSCRIPTS_DIR}/run.sh << EOF
#!/bin/bash -l
#SBATCH -D ./
#SBATCH -o ${OUTPUT_DIR}/${JOB_NAME}.%j
#SBATCH -e ${OUTPUT_DIR}/${JOB_NAME}.%j
#SBATCH -J ${JOB_NAME}
${JOB_TIME}
${JOB_QUEUE}
# Node feature:
#SBATCH --exclusive
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --ntasks-per-socket=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --threads-per-core=1

### Debug and Analytics
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1

### Modules and env variables
case \$(hostname) in
  cobra01) source cobra.env;;
  raven01) source raven.env;;
esac
source ${PROJ_ROOT}/.env
# export PROJ_ROOT=${PROJ_ROOT}
# export DATA_ROOT=${DATA_ROOT}

cp ${SUBSCRIPTS_DIR}/run.sh ${SUBSCRIPTS_DIR}/${JOB_NAME}.\${SLURM_JOB_ID}.sh

module list

echo -e "Nodes: \${SLURM_JOB_NUM_NODES} \t NTASK: \${SLURM_NTASKS}"
echo "\${SLURM_NODELIST}"

# Run the program:
${RUN} train.py \\
  --data_root ${DATA_ROOT}\\
	--num_epochs 1 \\
	${DATA_ARG}
	--shuffle  --num_workers ${NUM_WORKERS} --batch_size ${BS} \\
	--num_G_accumulations 1 --num_D_accumulations 1 \\
	--num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \\
	--G_attn 64 --D_attn 64 \\
	--G_nl inplace_relu --D_nl inplace_relu \\
	--SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \\
	--G_ortho 0.0 \\
	--G_shared \\
	--G_init ortho --D_init ortho \\
	--hier --dim_z 120 --shared_dim 128 \\
	--G_eval_mode \\
	--G_ch 96 --D_ch 96 \\
	--ema --use_ema --ema_start 20000 \\
	--test_every 2000 --save_every 1000 --num_best_copies 5 --num_save_copies 2 --seed 0 \\
	--use_multiepoch_sampler

EOF

echo "Submitting job ${JOB_NAME}"
echo "Submission script: ${SUBSCRIPTS_DIR}/run.sh"
sbatch ${SUBSCRIPTS_DIR}/run.sh
