#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
BASE_CKPT="${BASE_CKPT:-$PROJECT_ROOT/machine_unlearning/ckpts/compvis/style50/compvis.ckpt}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_ROOT/machine_unlearning/results/sequential}"
: "${STYLE_CLASSIFIER:?Set STYLE_CLASSIFIER to the style classifier checkpoint}"
: "${OBJECT_CLASSIFIER:?Set OBJECT_CLASSIFIER to the object classifier checkpoint}"
mkdir -p "$PROJECT_ROOT/machine_unlearning/slurm/logs"

for method in esd salun ediff; do
  prev=""
  for step in {1..6}; do
    dep=()
    [[ -z "$prev" ]] || dep=(--dependency="afterok:$prev")
    prev=$(sbatch --parsable "${dep[@]}" --export="ALL,PROJECT_ROOT,BASE_CKPT,RESULTS_ROOT,METHOD=$method,STEP=$step" "$PROJECT_ROOT/machine_unlearning/slurm/train_sequential.sbatch")
    echo "Submitted $method step $step as $prev"
  done
  case "$method" in
    esd) last_esd="$prev" ;;
    salun) last_salun="$prev" ;;
    ediff) last_ediff="$prev" ;;
  esac
done

eval_job=$(sbatch --parsable --dependency="afterok:$last_esd:$last_salun:$last_ediff" --export=ALL,PROJECT_ROOT,RESULTS_ROOT,STYLE_CLASSIFIER,OBJECT_CLASSIFIER "$PROJECT_ROOT/machine_unlearning/slurm/evaluate_sequential.sbatch")
echo "Submitted evaluation array as $eval_job"
