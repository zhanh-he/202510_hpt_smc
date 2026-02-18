#!/bin/bash
#SBATCH --job-name=scoreinf_ablate
#SBATCH --output=scoreinf_progress_%A_%a.log
#SBATCH --error=scoreinf_error_%A_%a.log
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --array=0-59
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhanh.he@research.uwa.edu.au

module load Anaconda3/2024.06 cuda/11.8 gcc/11.5.0
module list
source activate bark_env #hpt_mamba

echo "Running on host: $(hostname)"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
echo "SLURM ID: $SLURM_ARRAY_ID $SLURM_ARRAY_TASK_ID"

FOLDER_NAME=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
PROJECT_NAME=202510_hpt_smc
EXECUTABLE=$HOME/${PROJECT_NAME}
SCRATCH=$MYSCRATCH/${PROJECT_NAME}/$FOLDER_NAME
RESULTS=$MYGROUP/${PROJECT_NAME}_results/$FOLDER_NAME

mkdir -p $SCRATCH $RESULTS
echo SCRATCH is $SCRATCH
echo RESULTS dir is $RESULTS

echo "Copy path $EXECUTABLE to $SCRATCH"
cp -r $EXECUTABLE $SCRATCH
cd $SCRATCH/$PROJECT_NAME

WORKSPACE_DIR=$SCRATCH/$PROJECT_NAME/workspaces
mkdir -p $WORKSPACE_DIR

DATA_SRC=$MYSCRATCH/202510_hpt_data/workspaces/hdf5s
DATA_VIEW=$WORKSPACE_DIR/hdf5s
ln -s $DATA_SRC $DATA_VIEW

#############################################
# Adapter/method/loss ablation:
# 3 adapters x 5 methods x 4 losses = 60 array jobs.
ADAPTERS=("hpt" "hppnet" "dynest")
METHODS=("direct_output" "scrr" "dual_gated" "note_editor" "bilstm")
LOSSES=("velocity_bce" "velocity_mse" "kim_bce_l1" "score_inf_custom")

N_METHODS=${#METHODS[@]}
N_LOSSES=${#LOSSES[@]}
N_METHOD_LOSS=$((N_METHODS * N_LOSSES))

ADAPTER_IDX=$((SLURM_ARRAY_TASK_ID / N_METHOD_LOSS))
REM=$((SLURM_ARRAY_TASK_ID % N_METHOD_LOSS))
METHOD_IDX=$((REM / N_LOSSES))
LOSS_IDX=$((REM % N_LOSSES))

ADAPTER=${ADAPTERS[$ADAPTER_IDX]}
METHOD=${METHODS[$METHOD_IDX]}
LOSS_TYPE=${LOSSES[$LOSS_IDX]}

case "$ADAPTER" in
  hpt) MODEL_NAME="Single_Velocity_HPT" ;;
  hppnet) MODEL_NAME="HPPNet_SP" ;;
  dynest) MODEL_NAME="DynestAudioCNN" ;;
  *) echo "Unknown adapter: $ADAPTER"; exit 1 ;;
esac

echo "Adapter: $ADAPTER"
echo "Model  : $MODEL_NAME"
echo "Method : $METHOD"
echo "Loss   : $LOSS_TYPE"

python pytorch/train_score_inf.py \
  exp.workspace="$WORKSPACE_DIR" \
  model.name="$MODEL_NAME" \
  model.input2=null \
  model.input3=null \
  adapter.type="$ADAPTER" \
  score_informed.method="$METHOD" \
  loss.loss_type="$LOSS_TYPE"

#############################################
mv "$WORKSPACE_DIR/checkpoints/" "${RESULTS}/"
mv "$WORKSPACE_DIR/logs/" "${RESULTS}/"
cd $HOME
rm -r $SCRATCH # clean up the scratch space
source deactivate # Deactivate the conda environment - source or conda deactivate
echo scoreinf_ablate $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID finished at `date`
