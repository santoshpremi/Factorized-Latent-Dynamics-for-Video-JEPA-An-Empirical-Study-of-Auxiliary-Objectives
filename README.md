# Auxiliary Objectives for Video JEPA

This repository contains code for a small-scale empirical study of **18 auxiliary training objective variants** for Video Joint-Embedding Predictive Architecture (V-JEPA). We evaluate how these objectives affect transfer to motion-sensitive tasks (Diving-48, Something-Something V2) and appearance-sensitive tasks (ImageNet-100).

The most balanced variant in our mixed-dataset sweep, **FWM-HW-LD** (Factorized World-Model with Hard-Region-Weighted Latent Dynamics), factorizes the latent space into appearance and dynamics subspaces and applies dynamics prediction purely in latent space.

## Repository Structure

```
configs/            # YAML configs for the reported experiment variants
vjepa2/             # V-JEPA2 source code (modified train.py contains all auxiliary objectives)
scripts/            # Evaluation and data preparation scripts
results/            # Evaluation result JSONs for all experiments
```

The core contribution is in `vjepa2/app/vjepa_2_1/train.py`, where all auxiliary objectives are implemented as configurable additions to the base V-JEPA training loop.

## Installation

```bash
git clone https://github.com/santoshpremi/Factorized-Latent-Dynamics-for-Video-JEPA-An-Empirical-Study-of-Auxiliary-Objectives.git
cd Factorized-Latent-Dynamics-for-Video-JEPA-An-Empirical-Study-of-Auxiliary-Objectives
pip install -r vjepa2/requirements.txt
```

Requires Python 3.10+ and PyTorch 2.0+.

## Data Preparation

The scripts and YAML configs were run in the original cluster environment and contain absolute paths such as `/a/mm/VJEPA2/...`. Before running on a new machine, update the dataset CSV paths and output `folder` fields in the YAML configs.

Expected training CSV format:

```text
/absolute/path/to/video.mp4 label_id
```

**UCF-101:**
```bash
python scripts/prepare_ucf101.py
```

**Something-Something V2** (requires manual download from [Qualcomm](https://developer.qualcomm.com/software/ai-datasets/something-something)):
```bash
python scripts/prepare_ssv2.py
```

**ImageNet-100:** Subset of ImageNet-1K (100 classes). Use standard ImageNet download and the class list from the config.

**Diving-48:** Download from the [official source](http://www.svcl.ucsd.edu/projects/resound/dataset.html).

## Training

All reported training runs use 4 GPUs. Configs are in `configs/`. Run from the repository root after adapting paths inside the YAML file.

```bash
# Baseline (reference)
PYTHONPATH=vjepa2 python -m app.main \
    --fname configs/train_mixed_4gpu_baseline_seed42.yaml \
    --devices cuda:0 cuda:1 cuda:2 cuda:3

# FWM-HW-LD (factorized latent dynamics with hard-region weighting)
PYTHONPATH=vjepa2 python -m app.main \
    --fname configs/train_mixed_4gpu_fwm_hw_ld.yaml \
    --devices cuda:0 cuda:1 cuda:2 cuda:3

# Motion-Guided Masking
PYTHONPATH=vjepa2 python -m app.main \
    --fname configs/train_mixed_4gpu_motion_guided.yaml \
    --devices cuda:0 cuda:1 cuda:2 cuda:3
```

Checkpoints are saved to the `folder` specified in each YAML config.

### Available Configs

**Mixed-dataset experiments** (UCF-101 + SSv2 + ImageNet-100 pretraining; Diving-48 is evaluation only):

| Config | Objective |
|--------|-----------|
| `train_mixed_4gpu_baseline_seed42.yaml` | Baseline (reference) |
| `train_mixed_4gpu_baseline.yaml` | Baseline (additional run) |
| `train_mixed_4gpu_baseline_seed1337.yaml` | Baseline (additional seed) |
| `train_mixed_4gpu_motion_guided.yaml` | Motion-Guided Masking |
| `train_mixed_4gpu_hw.yaml` | Hard-Region Weighted Loss |
| `train_mixed_4gpu_delta.yaml` | Delta-Prediction |
| `train_mixed_4gpu_fwm.yaml` | Factorized World-Model |
| `train_mixed_4gpu_ld.yaml` | Latent Dynamics |
| `train_mixed_4gpu_fwm_ld.yaml` | FWM + Latent Dynamics |
| `train_mixed_4gpu_hw_ld.yaml` | HW + Latent Dynamics |
| `train_mixed_4gpu_fwm_hw_ld.yaml` | FWM + HW + Latent Dynamics |
| `train_mixed_4gpu_amg.yaml` | Aggressive Motion-Guided Masking |
| `train_mixed_4gpu_combo.yaml` | Combined (Delta + HW + Motion-Guided) |
| `train_mixed_4gpu_spectral.yaml` | Spectral Regularization |
| `train_mixed_4gpu_ltc.yaml` | Latent Temporal Contrastive |
| `train_mixed_4gpu_ac.yaml` | Action-Conditioned |
| `train_mixed_4gpu_ac_hw.yaml` | Action-Conditioned + HW |
| `train_mixed_4gpu_fac.yaml` | Factorized Action-Conditioned |

**UCF-101-only experiments** (kinematic and regularization variants):

| Config | Objective |
|--------|-----------|
| `train_ucf101_4gpu_baseline.yaml` | Baseline |
| `train_ucf101_4gpu_kinematic.yaml` | Kinematic Regularization |
| `train_ucf101_4gpu_kinematic_huber.yaml` | Kinematic (Huber loss) |
| `train_ucf101_4gpu_kinematic_accel.yaml` | Kinematic (acceleration) |
| `train_ucf101_4gpu_kinematic_split.yaml` | Kinematic (split head) |
| `train_ucf101_4gpu_kinematic_anneal.yaml` | Kinematic (annealed) |
| `train_ucf101_4gpu_sigreg.yaml` | SIGReg |
| `train_ucf101_4gpu_sigreg_no_ema.yaml` | SIGReg (no EMA) |
| `train_ucf101_4gpu_hamiltonian.yaml` | Hamiltonian |
| `train_ucf101_4gpu_velgate.yaml` | Velocity-Gated |
| `train_ucf101_4gpu_motion_guided.yaml` | Motion-Guided Masking |
| `train_ucf101_4gpu_future_predictive.yaml` | Future-Predictive |
| `train_ucf101_4gpu_motion_future.yaml` | Motion + Future |

## Evaluation

```bash
# Diving-48 (attentive probe, 48 classes)
PYTHONPATH=vjepa2 python scripts/eval_diving48.py \
    --config configs/train_mixed_4gpu_fwm_hw_ld.yaml \
    --checkpoint runs/train_mixed_4gpu_fwm_hw_ld/latest.pth.tar \
    --out_json results_new/fwm_hw_ld_diving48.json \
    --name FWM-HW-LD

# ImageNet-100 (linear probe, 100 classes)
PYTHONPATH=vjepa2 python scripts/eval_imagenet100.py \
    --config configs/train_mixed_4gpu_fwm_hw_ld.yaml \
    --checkpoint runs/train_mixed_4gpu_fwm_hw_ld/latest.pth.tar \
    --out_json results_new/fwm_hw_ld_imagenet100.json \
    --name FWM-HW-LD

# Something-Something V2 (attentive probe, 174 classes)
PYTHONPATH=vjepa2 python scripts/eval_ssv2.py \
    --config configs/train_mixed_4gpu_fwm_hw_ld.yaml \
    --checkpoint runs/train_mixed_4gpu_fwm_hw_ld/latest.pth.tar \
    --out_json results_new/fwm_hw_ld_ssv2.json \
    --name FWM-HW-LD
```

The evaluation scripts also contain original absolute dataset roots; update `base_dir` in each script for a new machine. Pre-computed evaluation results for all experiments are in `results/`.

## Key Results (Mixed-Dataset, Single Seed)

| Method | Diving-48 | ImageNet-100 | SSv2 |
|--------|-----------|--------------|------|
| Baseline (seed 42) | 8.68 | 24.86 | 8.39 |
| FWM-HW-LD | 8.38 | 30.78 | 11.60 |

All numbers are Top-1 accuracy (%). See `results/` for the full set.

For the UCF-101-only setting, Motion-Guided Masking improves all three reported evaluation metrics relative to the UCF-101 baseline: Diving-48 8.38 -> 8.68, ImageNet-100 12.02 -> 12.16, and SSv2 2.07 -> 3.45.

## Pre-trained Checkpoints

Checkpoints (~1.5 GB each) are not included in this repository. Available on request.

## Acknowledgments

Built on [V-JEPA](https://github.com/facebookresearch/vjepa) by Meta AI. Computing support from the Computer Vision Lab, CAIDAS & IFI, University of Würzburg, Germany.

## License

This repository follows the [MIT License](LICENSE). The underlying V-JEPA2 codebase is also released under the [MIT License](vjepa2/LICENSE).
