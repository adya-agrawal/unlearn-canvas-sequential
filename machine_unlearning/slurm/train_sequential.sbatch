#!/bin/bash
#SBATCH --job-name=uc_seq_train
#SBATCH --partition=pace-ice-gpu
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --output=machine_unlearning/slurm/logs/train_%j.out
#SBATCH --error=machine_unlearning/slurm/logs/train_%j.err

set -euo pipefail
module load anaconda3
conda activate "${CONDA_ENV:-$HOME/scratch/conda_envs/unlearn_canvas}"
export PYTHONNOUSERSITE=1 WANDB_MODE="${WANDB_MODE:-offline}"

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
BASE_CKPT="${BASE_CKPT:-$PROJECT_ROOT/machine_unlearning/ckpts/compvis/style50/compvis.ckpt}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_ROOT/machine_unlearning/results/sequential}"
REQUESTS=(Abstractionism Byzantine Cartoon Cold_Warm Ukiyoe Van_Gogh)

if [[ ! -f "$BASE_CKPT" || ! -f "$PROJECT_ROOT/data/quick-canvas-benchmark/Abstractionism/Architectures/1.jpg" ]]; then
  echo "Missing base checkpoint or training images. Set BASE_CKPT and PROJECT_ROOT." >&2
  exit 2
fi
if [[ ! "${METHOD:-}" =~ ^(esd|salun|ediff)$ || ! "${STEP:-}" =~ ^[1-6]$ ]]; then
  echo "Set METHOD=esd|salun|ediff and STEP=1..6." >&2
  exit 2
fi

idx=$((STEP - 1))
theme="${REQUESTS[$idx]}"
step_dir=$(printf '%s/%s/step_%02d' "$RESULTS_ROOT" "$METHOD" "$STEP")
input_ckpt="$BASE_CKPT"
if (( STEP > 1 )); then
  prev_dir=$(printf '%s/%s/step_%02d' "$RESULTS_ROOT" "$METHOD" "$((STEP - 1))")
  case "$METHOD" in
    esd) input_ckpt="$prev_dir/${REQUESTS[$((idx - 1))]}.pth" ;;
    salun) input_ckpt="$prev_dir/${REQUESTS[$((idx - 1))]}/sd_0.4.ckpt" ;;
    ediff) input_ckpt="$prev_dir/${REQUESTS[$((idx - 1))]}/sd.ckpt" ;;
  esac
fi
[[ -f "$input_ckpt" ]] || { echo "Missing input checkpoint: $input_ckpt" >&2; exit 3; }
mkdir -p "$step_dir"
echo "$METHOD step $STEP: $theme"

case "$METHOD" in
  esd)
    cd "$PROJECT_ROOT/machine_unlearning/mu_erasing_concept_esd"
    python stable_diffusion/train-scripts/train-esd-style.py --theme "$theme" --train_method xattn --ckpt_path "$input_ckpt" --config_path configs/train_esd.yaml --output_dir "$step_dir" --devices 0,0
    ;;
  salun)
    cd "$PROJECT_ROOT/machine_unlearning/mu_saliency_unlearn_salun"
    python generate_mask.py --theme "$theme" --output_dir "$step_dir" --ckpt_path "$input_ckpt" --config_path configs/train_salun.yaml --threshold 0.4
    python train-erase.py --theme "$theme" --output_dir "$step_dir" --input_dir "$step_dir" --ckpt_path "$input_ckpt" --config_path configs/train_salun.yaml --threshold 0.4 --train_method xattn
    ;;
  ediff)
    cd "$PROJECT_ROOT/machine_unlearning/mu_erase_diff_ediff"
    python train-erase.py --theme "$theme" --output_dir "$step_dir" --ckpt_path "$input_ckpt" --config_path configs/train_ediff.yaml --train_method xattn
    ;;
esac
