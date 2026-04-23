---
name: check-env
description: Use when `import lmdeploy` fails, CUDA not found, wrong Python version, editable install not recognized, or `conda run` invokes system Python instead of the env's Python.
---

# Check LMDeploy Dev Environment

> Last verified: 2026-04-23. Update Known Environments table if you add or rename envs.

## Configuration

| Placeholder    | Meaning                                                                      |
| -------------- | ---------------------------------------------------------------------------- |
| `<conda-root>` | Root of your miniconda/anaconda install (e.g. `/nvme1/zhouxinyu/miniconda3`) |
| `<env-name>`   | Conda env name (see Known Environments table below)                          |
| `<repo-dir>`   | Absolute path to the lmdeploy repo clone                                     |

## 1. Find and activate the conda env

```bash
conda env list                        # starred = currently active
conda activate <env-name>             # pick the right env for this project
```

## 2. Verify editable install

```bash
python -c "import lmdeploy; print(lmdeploy.__file__)"
# Must point into the repo dir, e.g. /nvme1/zhouxinyu/lmdeploy_vl/lmdeploy/__init__.py
```

If it doesn't:

```bash
pip install -e .                      # run from repo root
```

## 3. Pick a free GPU

```bash
nvidia-smi   # pick a GPU with 0% utilization and only a few MiB allocated
export CUDA_VISIBLE_DEVICES=<gpu_id>
```

## 4. Confirm python and CUDA

```bash
which python                          # must show conda env path, not /usr/bin/python
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())"
```

## Known Environments

| Repo dir       | Conda env | Direct python path                                |
| -------------- | --------- | ------------------------------------------------- |
| `lmdeploy_fp8` | `fp8`     | `/nvme1/zhouxinyu/miniconda3/envs/fp8/bin/python` |
| `lmdeploy_vl`  | `vl`      | `/nvme1/zhouxinyu/miniconda3/envs/vl/bin/python`  |

**Important**: `conda run -n <env> python` may invoke the system Python (3.6, no packages).
Always use the direct path for `pytest` and scripts:

```bash
CUDA_VISIBLE_DEVICES=X /nvme1/zhouxinyu/miniconda3/envs/<env>/bin/python -m pytest ...
```

## Troubleshooting

| Problem              | Fix                                                                  |
| -------------------- | -------------------------------------------------------------------- |
| `conda: not found`   | `source ~/miniconda3/etc/profile.d/conda.sh`                         |
| Wrong Python         | `conda deactivate && conda activate <env-name>`                      |
| `lmdeploy` not found | `pip install -e .` from repo root                                    |
| `conda run` wrong py | Use direct path: `/nvme1/zhouxinyu/miniconda3/envs/<env>/bin/python` |
