#!/bin/bash
#SBATCH --job-name=scoreinf_loss_ablate
#SBATCH --output=scoreinf_loss_progress_%A_%a.log
#SBATCH --error=scoreinf_loss_error_%A_%a.log
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --array=0-4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhanh.he@research.uwa.edu.au

module load Anaconda3/2024.06 cuda/11.8 gcc/11.5.0
module list
source activate bark_env

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
# Loss ablation on best model (set these three manually after model-selection):
BEST_ADAPTER="hpt"
BEST_MODEL_NAME="Single_Velocity_HPT"
BEST_METHOD="scrr"

LOSS_VARIANTS=(
  "score_inf_custom_huber_base"
  "score_inf_custom_huber_nodelta"
  "score_inf_custom_huber_bce1"
  "score_inf_custom_l1_base"
  "score_inf_custom_l1_only"
)

LOSS_VARIANT=${LOSS_VARIANTS[$SLURM_ARRAY_TASK_ID]}
echo "Adapter      : $BEST_ADAPTER"
echo "Model        : $BEST_MODEL_NAME"
echo "Method       : $BEST_METHOD"
echo "Loss Variant : $LOSS_VARIANT"

case "$LOSS_VARIANT" in
  score_inf_custom_huber_base)
    LOSS_ARGS=(
      loss.loss_type="score_inf_custom"
      loss.use_huber=true
      loss.w_huber=1.0
      loss.huber_delta=0.1
      loss.w_l1=1.0
      loss.w_bce=0.5
      loss.w_delta=0.01
    )
    ;;
  score_inf_custom_huber_nodelta)
    LOSS_ARGS=(
      loss.loss_type="score_inf_custom"
      loss.use_huber=true
      loss.w_huber=1.0
      loss.huber_delta=0.1
      loss.w_l1=1.0
      loss.w_bce=0.5
      loss.w_delta=0.0
    )
    ;;
  score_inf_custom_huber_bce1)
    LOSS_ARGS=(
      loss.loss_type="score_inf_custom"
      loss.use_huber=true
      loss.w_huber=1.0
      loss.huber_delta=0.1
      loss.w_l1=1.0
      loss.w_bce=1.0
      loss.w_delta=0.01
    )
    ;;
  score_inf_custom_l1_base)
    LOSS_ARGS=(
      loss.loss_type="score_inf_custom"
      loss.use_huber=false
      loss.w_huber=1.0
      loss.huber_delta=0.1
      loss.w_l1=1.0
      loss.w_bce=0.5
      loss.w_delta=0.01
    )
    ;;
  score_inf_custom_l1_only)
    LOSS_ARGS=(
      loss.loss_type="score_inf_custom"
      loss.use_huber=false
      loss.w_huber=1.0
      loss.huber_delta=0.1
      loss.w_l1=1.0
      loss.w_bce=0.0
      loss.w_delta=0.0
    )
    ;;
  *)
    echo "Unknown loss variant: $LOSS_VARIANT"
    exit 1
    ;;
esac

python pytorch/train_score_inf.py \
  exp.workspace="$WORKSPACE_DIR" \
  model.name="$BEST_MODEL_NAME" \
  model.input2=null \
  model.input3=null \
  adapter.type="$BEST_ADAPTER" \
  score_informed.method="$BEST_METHOD" \
  "${LOSS_ARGS[@]}"

#############################################
mv "$WORKSPACE_DIR/checkpoints/" "${RESULTS}/"
mv "$WORKSPACE_DIR/logs/" "${RESULTS}/"
cd $HOME
rm -r $SCRATCH
source deactivate
echo scoreinf_loss_ablate $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID finished at `date`
