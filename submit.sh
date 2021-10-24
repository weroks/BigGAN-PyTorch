#!/bin/bash -l

# Default values
DATASET='small_E256'
BS='34'
MODE='long'
NUM_WORKERS='0'
MONITORING="sys"
SEED='0'
G_LR='1e-4'
D_LR='4e-4'
D_STEPS="1"
ACCUM="1"
N_NODES='1'
N_TASKS='4'
DATA_ROOT="/ptmp/pierocor/datasets"
# OUT_ROOT="${PROJ_ROOT}/tmp"
OUT_ROOT="/ptmp/pierocor/hvd_out"

# DEFAULT TRAINING VALUES
TEST_EVERY='1000'
SAVE_EVERY='1000'
EMA_START='20000'

# OTHER VARS
ENV_VARS=''
ADD_ARGS=''
JOB_ARR_LENGTH='0'


print_usage() {
  printf "Usage: ./submit.sh
 -m <string> mode (test | long | test_arr | full) (i.e. 10 mins | 3h | array job 10m | array job 24h);
 -N <int> number of nodes;
 -n <int> number of gpus per node;
 -d <string> dataset (E256 | small_E256);
 -b <int> batch size (default 34);
 -a <int> accumulation (both G and D, default 1);
 -G <float> G learning rate (default 1e-4);
 -D <float> D learning rate (default 4e-4);
 -x <int> D training steps (default 1);
 -w <int> number of workers for DataLoader;
 -s <int> seed;
 -o <string> Output root directory;
 -e <string> Extra arguments;
 -t <string> Monitoring tool (sys | usr | pt);
 -i copy dataset in /dev/shm/
 -r resume from previous checkpoint.
 "
}

while getopts ':m:N:n:d:b:a:l:G:D:x:w:s:o:t:e:ri' flag; do
  case "${flag}" in
    m) MODE="${OPTARG}" ;;
    N) N_NODES="${OPTARG}" ;;
    n) N_TASKS="${OPTARG}" ;;
    d) DATASET="${OPTARG}" ;;
    b) BS="${OPTARG}" ;;
    a) ACCUM="${OPTARG}" ;;
    G) G_LR="${OPTARG}" ;;
    D) D_LR="${OPTARG}" ;;
    x) D_STEPS="${OPTARG}" ;;
    w) NUM_WORKERS="${OPTARG}" ;;
    s) SEED="${OPTARG}" ;;
    t) MONITORING="${OPTARG}" ;;
    o) OUT_ROOT="${OPTARG}" ;;
    e) ADD_ARGS="${ADD_ARGS} ${OPTARG}" ;;
    r) ADD_ARGS="${ADD_ARGS} --resume" ;;
    i) ADD_ARGS="${ADD_ARGS} --copy_in_mem" ;;
    *) print_usage
       exit 1 ;;
  esac
done

# PATHS
PROJ_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

WEIGHTS_ROOT="${OUT_ROOT}/weights"
LOGS_ROOT="${OUT_ROOT}/logs"
SAMPLES_ROOT="${OUT_ROOT}/samples/"
SUBSCRIPTS_DIR="${OUT_ROOT}/subscripts" 
OUTPUT_DIR="${OUT_ROOT}/output"

# Create outputs directories
mkdir -p ${SUBSCRIPTS_DIR}
mkdir -p ${OUTPUT_DIR}
mkdir -p ${WEIGHTS_ROOT}
mkdir -p ${LOGS_ROOT}
mkdir -p ${SAMPLES_ROOT}

JOB_NAME="h_${N_NODES}_${N_TASKS}_${DATASET}_${BS}x${ACCUM}_${G_LR}_${D_LR}_D${D_STEPS}_${MODE}_w${NUM_WORKERS}_s${SEED}"


case ${DATASET} in
  E256)
    DATA_ARG="--dataset E256_hdf5 \\" ;;
  small_E256)
    DATA_ARG="--dataset small_E256_hdf5 --load_in_mem \\" ;;
  ecoset_cs100)
    DATA_ARG="--dataset ecoset_cs100 \\" ;;
  ecoset_cs500)
    DATA_ARG="--dataset ecoset_cs500 \\" ;;
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

if [ "$N_TASKS" -gt "$NGPU" ]; then
  echo "Number of tasks per node ($N_TASKS) greater than number of GPUs ($NGPU)."
  exit 4
fi


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
    TEST_EVERY='10'
    SAVE_EVERY='15'
    EMA_START='17'
    N_EPOCHS='2'
    JOB_QUEUE=""
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

DATE=`date '+%d%m%Y%H%M%S'`

cat > ${SUBSCRIPTS_DIR}/run.sh.${DATE} << EOF
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
#SBATCH --nodes=${N_NODES}
#SBATCH --ntasks-per-socket=${NTASK_SOCKET}
#SBATCH --ntasks-per-node=${N_TASKS}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --threads-per-core=1
#SBATCH --signal=USR1@300

### Debug and Analytics
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1

### Modules and env variables
source ${PROJ_ROOT}/${SRC}


### store job submit script using a unique id:
if [ -z "\$SLURM_ARRAY_JOB_ID" ]; then
  cp ${SUBSCRIPTS_DIR}/run.sh.${DATE} ${SUBSCRIPTS_DIR}/${JOB_NAME}.\${SLURM_JOB_ID}.sh
elif [[ \${SLURM_ARRAY_TASK_ID} -eq 0 ]]; then
  cp ${SUBSCRIPTS_DIR}/run.sh.${DATE} ${SUBSCRIPTS_DIR}/${JOB_NAME}.\${SLURM_ARRAY_JOB_ID}_\${SLURM_ARRAY_TASK_ID}.sh
fi

### print modules and basic SLURM info
module list

echo -en "START:\t"
date '+%d/%m/%Y %H-%M-%S'
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
  --num_G_accumulations ${ACCUM} --num_D_accumulations ${ACCUM} \\
  --num_D_steps ${D_STEPS} --G_lr ${G_LR} --D_lr ${D_LR} --D_B2 0.999 --G_B2 0.999 \\
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
  --seed ${SEED} ${ADD_ARGS} &

### Wait python program
wait
echo "Batch script END!"
ls -atl /dev/shm/
EOF

echo "Submitting job ${JOB_NAME}"
echo "Submission script: ${SUBSCRIPTS_DIR}/run.sh.${DATE}"
echo "Output: ${OUTPUT_DIR}/${JOB_NAME}"
sbatch ${SUBSCRIPTS_DIR}/run.sh.${DATE}