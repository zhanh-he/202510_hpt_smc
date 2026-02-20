#!/bin/bash
#SBATCH --job-name=bark
#SBATCH --output=dyn_progress_%A_%a.log
#SBATCH --error=dyn_error_%A_%a.log
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --mem=24G
#SBATCH --array=0-5
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhanh.he@research.uwa.edu.au


# Load any required modules (if needed) - module load cuda/11.8 gcc/9.4.0
module load Anaconda3/2024.06 cuda/12.4.1 gcc/12.4.0
# v100: 12-16cpus, mem16G-32G

# leave in, it lists the environment loaded by the modules - https://wandb.ai/authorize
module list

# Activate the conda environment, may need "conda init"
# source ~/miniconda3/etc/profile.d/conda.sh
source activate bark_env
# conda activate hpt_mamba

# Print some useful information, Note: SLURM_JOBID is a unique number for every job.
echo "Running on host: $(hostname)"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
echo "SLURM ID: $SLURM_ARRAY_ID $SLURM_ARRAY_TASK_ID"

# Make W&B monitor exactly the GPUs exposed to this job
export WANDB__SERVICE__GPU_MONITOR_POLICY=visible
export WANDB__SERVICE__GPU_MONITOR_DEVICES="$CUDA_VISIBLE_DEVICES"

# Torch Distributed defaults for single-node multi-GPU runs
export GPUS_PER_NODE=2
export MASTER_ADDR=${MASTER_ADDR:-$(hostname)}
export MASTER_PORT=${MASTER_PORT:-$((12000 + RANDOM % 20000))}

# These are generic variables
FOLDER_NAME=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
EXECUTABLE=$HOME/202511_bark
SCRATCH=$MYSCRATCH/202511_bark/$FOLDER_NAME
RESULTS=$MYGROUP/202511_bark_result/$FOLDER_NAME 

###############################################
# Creates a unique directory in the SCRATCH directory for this job to run in.
if [ ! -d $SCRATCH ]; then
    mkdir -p $SCRATCH
fi
echo SCRATCH is $SCRATCH

###############################################
# Creates a unique directory in your GROUP directory for the results of this job
if [ ! -d $RESULTS ]; then
    mkdir -p $RESULTS
fi
echo the results directory is $RESULTS

#############################################
# Copy input files to $SCRATCH, then change directory to $SCRATCH
echo "Copy path $EXECUTABLE to $SCRATCH"
cp -r $EXECUTABLE $SCRATCH
cd $SCRATCH/202511_bark

#############################################
# link the dataset to real data folder
ln -s $MYSCRATCH/workspaces/hdf5s $SCRATCH/202511_bark/workspaces/hdf5s


#############################################
# Ablation study
# Shared: AdamW, fc_dropout=0.2, FPS=50, segment=60s
#############################################

LAUNCHER="torchrun --standalone --nnodes=1 --nproc_per_node=${GPUS_PER_NODE}"
BASE_CMD="$LAUNCHER pytorch/train.py exp.use_fsdp=true"

declare -a EXPERIMENTS=(
  "$BASE_CMD wandb.note='kaya' dataset.use_dataset=\"mazurka\" exp.model_name=SingleCNN exp.targets='[\"beat\"]' feature.audio_feature=\"sone\""
  "$BASE_CMD wandb.note='kaya' dataset.use_dataset=\"mazurka\" exp.model_name=SingleCNN exp.targets='[\"beat\"]' feature.audio_feature=\"logmel128\""
  "$BASE_CMD wandb.note='kaya' dataset.use_dataset=\"mazurka\" exp.model_name=SingleCNN exp.targets='[\"beat\"]' feature.audio_feature=\"logmel229\""

  "$BASE_CMD wandb.note='kaya' dataset.use_dataset=\"mazurka\" exp.model_name=SingleCNN exp.targets='[\"downbeat\"]' feature.audio_feature=\"sone\""
  "$BASE_CMD wandb.note='kaya' dataset.use_dataset=\"mazurka\" exp.model_name=SingleCNN exp.targets='[\"downbeat\"]' feature.audio_feature=\"logmel128\""
  "$BASE_CMD wandb.note='kaya' dataset.use_dataset=\"mazurka\" exp.model_name=SingleCNN exp.targets='[\"downbeat\"]' feature.audio_feature=\"logmel229\""
  # learning rate sweep at latent_dim=32
  # "python pytorch/train.py wandb.note='ablation mtcnn lr1e-3' exp.model_name=MultiTaskCNN exp.use_adamw=true exp.learning_rate=1e-3 cnn.latent_dim=32"
  # "python pytorch/train.py wandb.note='ablation mtcnn lr3e-4' exp.model_name=MultiTaskCNN exp.use_adamw=true exp.learning_rate=3e-4 cnn.latent_dim=32"
  # "python pytorch/train.py wandb.note='ablation mtcnn lr1e-4' exp.model_name=MultiTaskCNN exp.use_adamw=true exp.learning_rate=1e-4 cnn.latent_dim=32"
)

CMD="${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"
echo "Running: $CMD"
eval $CMD

#############################################
#   $OUTPUT file to the unique results dir
# note this can be a copy or move
mv ./workspaces/checkpoints/ ${RESULTS}/

cd $HOME

###########################
# Clean up $SCRATCH

rm -r $SCRATCH

# Deactivate the conda environment - source or conda deactivate
source deactivate
# conda deactivate

echo bark $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID finished at `date`
