#!/bin/bash
#SBATCH --job-name=uc_seq_eval
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --output=machine_unlearning/slurm/logs/uc_seq_eval_%A_%a.out
#SBATCH --error=machine_unlearning/slurm/logs/uc_seq_eval_%A_%a.err
#SBATCH --array=0-17%1

set -euo pipefail

module load anaconda3
conda activate "${CONDA_ENV:-$HOME/scratch/conda_envs/unlearn_canvas}"
export PYTHONNOUSERSITE=1
export WANDB_MODE="${WANDB_MODE:-offline}"

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/machine_unlearning/results/sequential}"
EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/machine_unlearning/evaluation/eval_results/sequential}"
STYLE_CLASSIFIER="${STYLE_CLASSIFIER:-}"
OBJECT_CLASSIFIER="${OBJECT_CLASSIFIER:-}"
SEEDS=(188 288 588 688 888)
REQUESTS=(Abstractionism Byzantine Cartoon Cold_Warm Ukiyoe Van_Gogh)
METHODS=(esd salun ediff)

if [[ -z "$STYLE_CLASSIFIER" || -z "$OBJECT_CLASSIFIER" || ! -f "$STYLE_CLASSIFIER" || ! -f "$OBJECT_CLASSIFIER" ]]; then
  echo "Set STYLE_CLASSIFIER and OBJECT_CLASSIFIER to the downloaded UnlearnCanvas classifier checkpoints." >&2
  exit 2
fi

task_id="${SLURM_ARRAY_TASK_ID}"
method_idx=$((task_id / 6))
step_idx=$((task_id % 6))
method="${METHODS[$method_idx]}"
step=$((step_idx + 1))
theme="${REQUESTS[$step_idx]}"
step_dir=$(printf '%s/%s/step_%02d' "$RESULTS_ROOT" "$method" "$step")
case "$method" in
  esd) ckpt="$step_dir/${theme}.pth" ;;
  salun) ckpt="$step_dir/${theme}/sd_0.4.ckpt" ;;
  ediff) ckpt="$step_dir/${theme}/sd.ckpt" ;;
esac
if [[ ! -f "$ckpt" ]]; then
  echo "Checkpoint not found: $ckpt" >&2
  exit 3
fi

case "$method" in
  esd) sampler="$PROJECT_ROOT/machine_unlearning/evaluation/sampling_unlearned_models/esd.py" ;;
  salun) sampler="$PROJECT_ROOT/machine_unlearning/evaluation/sampling_unlearned_models/salun.py" ;;
  ediff) sampler="$PROJECT_ROOT/machine_unlearning/evaluation/sampling_unlearned_models/ediff.py" ;;
esac
sampler_lib="$PROJECT_ROOT/machine_unlearning/mu_erasing_concept_esd"
image_dir="$EVAL_ROOT/$method/step_$(printf '%02d' "$step")/images"
accuracy_dir="$EVAL_ROOT/$method/step_$(printf '%02d' "$step")/accuracy"
mkdir -p "$image_dir" "$accuracy_dir"

cd "$PROJECT_ROOT/machine_unlearning/evaluation"
export PYTHONPATH="$PROJECT_ROOT/machine_unlearning/evaluation:$sampler_lib:$sampler_lib/stable_diffusion:${PYTHONPATH:-}"
python "$sampler" --theme "sequential_step_${step}" --ckpt "$ckpt" --output_dir "$image_dir" --config "$sampler_lib/configs/generate_sd.yaml" --seed "${SEEDS[@]}"

python quantitative/accuracy.py --input_dir "$image_dir" --output_dir "$accuracy_dir/style" --ckpt "$STYLE_CLASSIFIER" --task style --seed "${SEEDS[@]}"
python quantitative/accuracy.py --input_dir "$image_dir" --output_dir "$accuracy_dir/object" --ckpt "$OBJECT_CLASSIFIER" --task class --seed "${SEEDS[@]}"

echo "Completed $method step $step ($theme): $accuracy_dir"
