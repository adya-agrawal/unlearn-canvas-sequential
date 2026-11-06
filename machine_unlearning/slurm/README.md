# Sequential unlearning on PACE-ICE

The submit script chains six requests (Abstractionism, Byzantine, Cartoon, Cold_Warm, Ukiyoe, Van_Gogh) for ESD, SalUn, and EDiff, then submits evaluation. Each request is a separate job to fit ICE's walltime. Paths default to `~/scratch`; SLURM uses `ice-gpu` and H100. Check `sinfo` on ICE if your available partition/GPU differs.

Run these commands from the repository root on ICE. Keep package caches and large assets on scratch; the 30 GB home quota is easy to exceed.

## Environment

```bash
module load anaconda3
mkdir -p "$HOME/scratch/conda_pkgs" "$HOME/scratch/conda_tmp" "$HOME/scratch/conda_envs"
export CONDA_PKGS_DIRS="$HOME/scratch/conda_pkgs" TMPDIR="$HOME/scratch/conda_tmp"
conda env create -f environment.yaml -p "$HOME/scratch/conda_envs/unlearn_canvas"
conda activate "$HOME/scratch/conda_envs/unlearn_canvas"
pip install wandb timm==0.9.16
```

## Downloaded assets

The run used three checkpoints from Hugging Face's `NormalUhr/UnlearnCanvas-ckpts` repository and the UnlearnCanvas image dataset from `OPTML-Group/UnlearnCanvas`. The dataset repository is stored as Parquet shards; the training scripts need the extracted JPG directory tree.

```bash
ASSET_ROOT="$HOME/scratch/unlearncanvas-assets"
mkdir -p "$ASSET_ROOT/diffusion/compvis" "$ASSET_ROOT/cls_model"

# Stable Diffusion checkpoint (~7.2 GB)
curl -fL --retry 5 --retry-all-errors -C - \
  "https://huggingface.co/NormalUhr/UnlearnCanvas-ckpts/resolve/main/diffusion/compvis/compvis.ckpt" \
  -o "$ASSET_ROOT/diffusion/compvis/compvis.ckpt"

# Style and object classifier checkpoints (~3.4 GB each)
curl -fL --retry 5 --retry-all-errors -C - \
  "https://huggingface.co/NormalUhr/UnlearnCanvas-ckpts/resolve/main/cls_model/style50.pth" \
  -o "$ASSET_ROOT/cls_model/style50.pth"
curl -fL --retry 5 --retry-all-errors -C - \
  "https://huggingface.co/NormalUhr/UnlearnCanvas-ckpts/resolve/main/cls_model/style50_cls.pth" \
  -o "$ASSET_ROOT/cls_model/style50_cls.pth"
```

The dataset used for this run was downloaded from [the Hugging Face dataset repository](https://huggingface.co/datasets/OPTML-Group/UnlearnCanvas) and extracted to `$HOME/scratch/unlearn-canvas-sequential/data/quick-canvas-benchmark` (24,400 JPGs). The HF files are Parquet shards, not a ready-made JPG directory. Download the shards with:

```bash
mkdir -p "$HOME/scratch/hf_unlearncanvas"
huggingface-cli download OPTML-Group/UnlearnCanvas --repo-type dataset \
  --include 'data/train-*.parquet' --local-dir "$HOME/scratch/hf_unlearncanvas"
```

Each Parquet row contains `image.bytes` plus a caption such as `An Architectures image in Abstractionism style`; extract the bytes into `STYLE/OBJECT/image_idx.jpg`. A streaming `datasets.load_dataset` call did not work in the ICE environment. The extracted tree for this run was `$HOME/scratch/unlearn-canvas-sequential/data/quick-canvas-benchmark`.

This command converts the downloaded shards into that directory layout (requires `pyarrow`, which is in the run environment):

```bash
DATA="$HOME/scratch/unlearn-canvas-sequential/data/quick-canvas-benchmark"
DATA="$DATA" python -c 'import glob, os, re
from pathlib import Path
import pyarrow.parquet as pq
out = Path(os.environ["DATA"])
for shard in sorted(glob.glob(str(Path.home() / "scratch/hf_unlearncanvas/data/train-*.parquet"))):
    for batch in pq.ParquetFile(shard).iter_batches(columns=["image", "text"], batch_size=64):
        for row in batch.to_pylist():
            match = re.fullmatch(r"An? (.+?) image in (.+?) style\.?", row["text"].strip(), re.I)
            if not match: raise ValueError("Unrecognized caption: " + row["text"])
            obj, style = match.groups()
            target = out / style / obj / Path(row["image"]["path"]).name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(row["image"]["bytes"])'
```

The HF export uses spaces in some style directory names and calls the photo-realistic folder `photo`; this checkout expects underscores and `Seed_Images`. Create aliases without duplicating the images:

```bash
DATA="$HOME/scratch/unlearn-canvas-sequential/data/quick-canvas-benchmark"
for d in "$DATA"/*\ *; do
  [ -d "$d" ] || continue
  name=${d##*/}
  alias=${name// /_}
  [ -e "$DATA/$alias" ] || ln -s "$name" "$DATA/$alias"
done
[ -e "$DATA/Seed_Images" ] || ln -s photo "$DATA/Seed_Images"
test -f "$DATA/Seed_Images/Architectures/1.jpg" && echo "Dataset ready"
find "$DATA" -type f -name '*.jpg' | wc -l
```

## Submit training and evaluation

Set the asset paths, then submit from the repository root. The ICE scripts use `ice-gpu` and request an H100; check `sinfo -h -o "%P %G"` if the partition or GPU names change.

```bash
export PROJECT_ROOT="$PWD" CONDA_ENV="$HOME/scratch/conda_envs/unlearn_canvas"
export BASE_CKPT="$HOME/scratch/unlearncanvas-assets/diffusion/compvis/compvis.ckpt"
export STYLE_CLASSIFIER="$HOME/scratch/unlearncanvas-assets/cls_model/style50.pth"
export OBJECT_CLASSIFIER="$HOME/scratch/unlearncanvas-assets/cls_model/style50_cls.pth"
bash machine_unlearning/slurm/submit_sequential.sh
```

The submit script chains all six requests for ESD, SalUn, and EDiff, then submits an 18-task evaluation array after the training chains succeed. Check progress and finished job status with:

```bash
squeue -u "$USER"
sacct -u "$USER" --format=JobID,JobName,State,ExitCode,Elapsed
find machine_unlearning/evaluation/eval_results/sequential -name result.pth -print
```

After all evaluation tasks finish, create the summary CSV:

```bash
cd "$PROJECT_ROOT/machine_unlearning/evaluation"
python quantitative/summarize_sequential.py
```

The CSV is `machine_unlearning/evaluation/eval_results/sequential/table_a5_reproduction.csv`; accuracy results are also saved per method and step under `eval_results/sequential/<method>/step_XX/accuracy/`.
