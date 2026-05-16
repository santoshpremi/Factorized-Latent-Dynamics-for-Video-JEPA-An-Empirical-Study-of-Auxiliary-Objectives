# Auxiliary Objectives for Video JEPA

This repository contains code for a systematic empirical study of **18 auxiliary training objectives** for Video Joint-Embedding Predictive Architecture (V-JEPA). We evaluate how each objective affects transfer to motion-sensitive tasks (Diving-48, Something-Something V2) and appearance-sensitive tasks (ImageNet-100).

The best-performing variant, **FWM-HW-LD** (Factorized World-Model with Hard-Region-Weighted Latent Dynamics), factorizes the latent space into appearance and dynamics subspaces and applies dynamics prediction purely in latent space.

## Repository Structure

```
configs/            # YAML configs for all 18 experiment variants
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

**UCF-101:**
```bash
python scripts/download_ucf101.py --output_dir data/ucf101
python scripts/prepare_ucf101.py --root data/ucf101
```

**Something-Something V2** (requires manual download from [Qualcomm](https://developer.qualcomm.com/software/ai-datasets/something-something)):
```bash
python scripts/prepare_ssv2.py --root data/ssv2
```

**ImageNet-100:** Subset of ImageNet-1K (100 classes). Use standard ImageNet download and the class list from the config.

**Diving-48:** Download from the [official source](http://www.svcl.ucsd.edu/projects/resound/dataset.html).

## Training

All experiments use 4 GPUs. Configs are in `configs/`.

```bash
# Baseline (reference)
torchrun --nproc_per_node=4 -m app.vjepa_2_1.train \
    --fname configs/train_mixed_4gpu_baseline_seed42.yaml

# FWM-HW-LD (factorized latent dynamics with hard-region weighting)
torchrun --nproc_per_node=4 -m app.vjepa_2_1.train \
    --fname configs/train_mixed_4gpu_fwm_hw_ld.yaml

# Motion-Guided Masking
torchrun --nproc_per_node=4 -m app.vjepa_2_1.train \
    --fname configs/train_mixed_4gpu_motion_guided.yaml
```

Run from the repository root. Checkpoints are saved to `runs/<config_name>/`.

### Available Configs

**Mixed-dataset experiments** (UCF-101 + SSv2 + Diving-48 + ImageNet-100):

| Config | Objective |
|--------|-----------|
| `train_mixed_4gpu_baseline_seed42.yaml` | Baseline (reference) |
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
python scripts/eval_diving48.py \
    --config configs/train_mixed_4gpu_fwm_hw_ld.yaml \
    --checkpoint runs/train_mixed_4gpu_fwm_hw_ld/latest.pth.tar

# ImageNet-100 (linear probe, 100 classes)
python scripts/eval_imagenet100.py \
    --config configs/train_mixed_4gpu_fwm_hw_ld.yaml \
    --checkpoint runs/train_mixed_4gpu_fwm_hw_ld/latest.pth.tar

# Something-Something V2 (attentive probe, 174 classes)
python scripts/eval_ssv2.py \
    --config configs/train_mixed_4gpu_fwm_hw_ld.yaml \
    --checkpoint runs/train_mixed_4gpu_fwm_hw_ld/latest.pth.tar
```

Pre-computed evaluation results for all experiments are in `results/`.

## Key Results (Mixed-Dataset, Single Seed)

| Method | Diving-48 | ImageNet-100 | SSv2 |
|--------|-----------|--------------|------|
| Baseline (seed 42) | 8.68 | 24.86 | 8.39 |
| Motion-Guided | -- | 19.70 | 6.19 |
| FWM-HW-LD | 8.38 | 30.78 | 11.60 |

All numbers are Top-1 accuracy (%). See `results/` for the full set.

## Pre-trained Checkpoints

Checkpoints (~1.5 GB each) are not included in this repository. Available on request.

## Acknowledgments

Built on [V-JEPA](https://github.com/facebookresearch/vjepa) by Meta AI. Computing support from the Computer Vision Lab, CAIDAS & IFI, University of Würzburg, Germany.

## License

The V-JEPA2 codebase is released under the [MIT License](vjepa2/LICENSE). Our modifications follow the same license.
