# Sequential unlearning on PACE-ICE

The submit script chains six requests (Abstractionism, Byzantine, Cartoon, Cold_Warm, Ukiyoe, Van_Gogh) for ESD, SalUn, and EDiff, then submits evaluation. Each request is a separate job to fit ICE's walltime. Paths default to `~/scratch`; SLURM uses `pace-ice-gpu` and H100. Check `sinfo` on ICE if your available partition/GPU differs.

Create the environment and keep Conda's cache off your quota-limited home directory:

```bash
module load anaconda3
mkdir -p "$HOME/scratch/conda_pkgs" "$HOME/scratch/conda_tmp" "$HOME/scratch/conda_envs"
export CONDA_PKGS_DIRS="$HOME/scratch/conda_pkgs" TMPDIR="$HOME/scratch/conda_tmp"
conda env create -f environment.yaml -p "$HOME/scratch/conda_envs/unlearn_canvas"
conda activate "$HOME/scratch/conda_envs/unlearn_canvas"
pip install wandb timm==0.9.16
```

Set the checkpoint and classifier paths, then submit from the repo root. The images should be at `data/quick-canvas-benchmark` and the base model at `machine_unlearning/ckpts/compvis/style50/compvis.ckpt`.

```bash
export PROJECT_ROOT="$PWD" CONDA_ENV="$HOME/scratch/conda_envs/unlearn_canvas"
export BASE_CKPT="$PROJECT_ROOT/machine_unlearning/ckpts/compvis/style50/compvis.ckpt"
export STYLE_CLASSIFIER=/path/to/style_classifier.pth OBJECT_CLASSIFIER=/path/to/object_classifier.pth
bash machine_unlearning/slurm/submit_sequential.sh
```

After the evaluation array completes, run `cd machine_unlearning/evaluation && python quantitative/summarize_sequential.py`. It writes `eval_results/sequential/table_a5_reproduction.csv` with per-request UA and retained-concept RA.
