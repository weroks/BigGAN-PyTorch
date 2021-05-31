#!/bin/bash -l

# Default values

DATASET='small_E256'
BS='34'
MODE='long'
NUM_WORKERS='8'
MONITORING="sys"
PARALLEL="false"
SEED='0'


# DEFAULT TRAINING VALUES
TEST_EVERY='1000'
SAVE_EVERY='1000'
EMA_START='20000'
G_LR='1e-4'
D_LR='4e-4'

# OTHER VARS
ENV_VARS=''
ADD_ARGS=''
JOB_ARR_LENGTH='0'

# PATHS
DATA_ROOT="/ptmp/pierocor/datasets/"
OUT_ROOT="/ptmp/pierocor/BigGan_out/"
WEIGHTS_ROOT="${OUT_ROOT}/weights/"
LOGS_ROOT="${OUT_ROOT}/logs/"
SAMPLES_ROOT="${OUT_ROOT}/samples/"

PROJ_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SUBSCRIPTS_DIR="${OUT_ROOT}/subscripts/" 
OUTPUT_DIR="${OUT_ROOT}/output/"

# Create outputs directories
mkdir -p ${SUBSCRIPTS_DIR}
mkdir -p ${OUTPUT_DIR}
mkdir -p ${WEIGHTS_ROOT}
mkdir -p ${LOGS_ROOT}
mkdir -p ${SAMPLES_ROOT}


print_usage() {
  printf "Usage: ./submit.sh
 -m <string> mode (test | long | test_arr | full) (i.e. 10 mins | 3h | array job 10m | array job 24h);
 -d <string> dataset (E256 | small_E256);
 -b <int> batch size;
 -w <int> number of workers for DataLoader;
 -s <int> seed;
 -t <string> Monitoring tool (sys | usr | pt);
 -p Use all GPUs with DataParallel;
 -r resume from previous checkpoint.
 "
}

while getopts ':m:d:b:w:s:t:pr' flag; do
  case "${flag}" in
    m) MODE="${OPTARG}" ;;
    d) DATASET="${OPTARG}" ;;
    b) BS="${OPTARG}" ;;
    w) NUM_WORKERS="${OPTARG}" ;;
    s) SEED="${OPTARG}" ;;
    t) MONITORING="${OPTARG}" ;;
    p) PARALLEL="true" ;;
    r) ADD_ARGS="${ADD_ARGS} --resume" ;;
    *) print_usage
       exit 1 ;;
  esac
done

JOB_NAME="${DATASET}_${BS}_${MODE}_w${NUM_WORKERS}_${G_LR}_${D_LR}_s${SEED}"

case ${DATASET} in
  E256)
    DATA_ARG="--dataset E256_hdf5 \\" ;;
  small_E256)
    DATA_ARG="--dataset small_E256_hdf5 --load_in_mem \\" ;;
  *)
    echo "Wrong dataset name"
    exit 1
    ;;
esac

case $MONITORING in
  sys) RUN="srun python" ;;
  usr) RUN="export HPCMD_SLEEP=0; export HPCMD_AWAKE=10; srun hpcmd_slurm python" ;;
  pt) RUN="srun hpcmd_suspend python -m torch.utils.bottleneck" ;;
  *) echo "Worng monitoring tool. Available: sys, usr, pt."
     exit 2 ;;
esac

case $PARALLEL in
  true)
    ADD_ARGS="${ADD_ARGS} --parallel"
    JOB_NAME="${JOB_NAME}_p" ;;
  false)
    ENV_VARS="${ENV_VARS}
export CUDA_VISIBLE_DEVICES=0" ;;
esac

case $(hostname) in
  cobra01)
    SRC="cobra.env"
    GPU="v100"
    NGPU="2"
    CPUS="20"
    NTASK_SOCKET="1"
    TEST_Q="gpudev"
    ;;
  raven01|raven02)
    SRC="raven.env"
    GPU="a100"
    CPUS="18"
    NGPU="4"
    NTASK_SOCKET="2"
    TEST_Q="test"
    ;;
  *)
    echo "Host not recognized. Available: raven01, cobra01."
    exit 3 ;;
esac

case ${MODE} in
  test)
    N_EPOCHS='1'
    JOB_QUEUE="#SBATCH -p ${TEST_Q}"
    JOB_TIME='#SBATCH --time=0:10:00'
    JOB_NAME_EXT="%j"
    ;;
  long)
    N_EPOCHS='1'
    JOB_QUEUE=''
    JOB_TIME='#SBATCH --time=03:00:00'
    JOB_NAME_EXT="%j"
    ;;
  test_arr)
    TEST_EVERY='120'
    SAVE_EVERY='120'
    EMA_START='200'
    N_EPOCHS='3'
    JOB_QUEUE=''
    JOB_TIME='#SBATCH --time=0-00:10:00'
    JOB_ARRAY="#SBATCH --array=0-${JOB_ARR_LENGTH}%1"
    JOB_NAME_EXT="%A_%a"
    RESUME_CHECKPOINTS="
if [ \$SLURM_ARRAY_TASK_ID -ne 0 ]; then
  RESUME=\"--resume\"
fi
"
    ADD_ARGS="${ADD_ARGS} \${RESUME}"
    ;;
  full)
    N_EPOCHS='100'
    JOB_QUEUE=''
    JOB_TIME='#SBATCH --time=1-00:00:00'
    JOB_ARRAY="#SBATCH --array=0-${JOB_ARR_LENGTH}%1"
    JOB_NAME_EXT="%A_%a"
    RESUME_CHECKPOINTS="
if [ \$SLURM_ARRAY_TASK_ID -ne 0 ]; then
  RESUME=\"--resume\"
fi
"
    ADD_ARGS="${ADD_ARGS} \${RESUME}"
    ;;
  *)
    echo "Wrong mode: long, test, huge."
esac


cat > ${SUBSCRIPTS_DIR}/run.sh << EOF
#!/bin/bash -l
#SBATCH -D ./
#SBATCH -o ${OUTPUT_DIR}/${JOB_NAME}.${JOB_NAME_EXT}
#SBATCH -e ${OUTPUT_DIR}/${JOB_NAME}.${JOB_NAME_EXT}
#SBATCH -J ${JOB_NAME}
${JOB_TIME}
${JOB_QUEUE}
${JOB_ARRAY}
# Node feature:
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:${GPU}:${NGPU}
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --ntasks-per-socket=${NTASK_SOCKET}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --threads-per-core=1

### Debug and Analytics
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1

### Modules and env variables
source ${PROJ_ROOT}/${SRC}
${ENV_VARS}

### store job submit script using a unique id:
if [ -z "\$SLURM_ARRAY_JOB_ID" ]; then
  cp ${SUBSCRIPTS_DIR}/run.sh ${SUBSCRIPTS_DIR}/${JOB_NAME}.\${SLURM_JOB_ID}.sh
elif [[ \${SLURM_ARRAY_TASK_ID} -eq 0 ]]; then
  cp ${SUBSCRIPTS_DIR}/run.sh ${SUBSCRIPTS_DIR}/${JOB_NAME}.\${SLURM_ARRAY_JOB_ID}_\${SLURM_ARRAY_TASK_ID}.sh
fi

### print modules and basic SLURM info
module list

echo -e "Nodes: \${SLURM_JOB_NUM_NODES} \t NTASK: \${SLURM_NTASKS}"
echo "\${SLURM_NODELIST}"

### RESUME variable defined for job arrays only, if not empty
${RESUME_CHECKPOINTS}

### Run the program:
${RUN} train.py \\
  --data_root ${DATA_ROOT} \\
  --weights_root ${WEIGHTS_ROOT} \\
  --logs_root ${LOGS_ROOT} \\
  --samples_root ${SAMPLES_ROOT} \\
  --num_epochs ${N_EPOCHS} \\
  ${DATA_ARG}
  --shuffle  --num_workers ${NUM_WORKERS} --batch_size ${BS} \\
  --num_G_accumulations 1 --num_D_accumulations 1 \\
  --num_D_steps 1 --G_lr ${G_LR} --D_lr ${D_LR} --D_B2 0.999 --G_B2 0.999 \\
  --G_attn 64 --D_attn 64 \\
  --G_nl inplace_relu --D_nl inplace_relu \\
  --SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \\
  --G_ortho 0.0 \\
  --G_shared \\
  --G_init ortho --D_init ortho \\
  --hier --dim_z 120 --shared_dim 128 \\
  --G_eval_mode \\
  --G_ch 96 --D_ch 96 \\
  --ema --use_ema --ema_start ${EMA_START} \\
  --test_every ${TEST_EVERY} --save_every ${SAVE_EVERY} \\
  --num_best_copies 5 --num_save_copies 2 \\
  --seed ${SEED} \\
  --use_multiepoch_sampler ${ADD_ARGS}

EOF

echo "Submitting job ${JOB_NAME}"
echo "Submission script: ${SUBSCRIPTS_DIR}/run.sh"
sbatch ${SUBSCRIPTS_DIR}/run.sh
