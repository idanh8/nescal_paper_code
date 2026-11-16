# NeSCal

## Overview

NeSCal is a training regularizer for neural network calibration. It encourages similar predictive distributions for neighbouring examples in embedding space, using a nearest-neighbour graph constructed after an initial training period.

This repository accompanies *Neighborhood Smoothing for Calibration* by Idan Horowitz and Avigdor Gal.

[Read the paper](https://arxiv.org/abs/2610.09020)

## Main findings

- Evaluated across 11 image, text, and tabular dataset–architecture settings.
- Lower test negative log-likelihood than cross-entropy training in all 11 main settings.
- After temperature scaling, achieved the lowest negative log-likelihood among evaluated training methods in seven of eight image/tabular settings.
- Introduces no additional inference-time computation.
  
## Setup

```bash
conda env create -f environment.yml
conda activate nescal
```

Install PyTorch and torchvision for your CUDA setup, then run:

```bash
python -m pip install -e .
```

Run all commands from the repository root.

## Run experiments

Choose or edit a configuration in `configs/examples/`: `image.json`, `tabular.json`, `gpcnn.json`, `frozen.json` or `synthetic.json`. These contain source defaults; set the dataset, architecture, training budget and search ranges for your experiment. No author checkpoints or manifests are required. Frozen-encoder runs prepare their own features and perform encoder fine-tuning when configured.

Run the configured NeSCal and competitor sweeps, post-hoc calibration and reporting:

```bash
python scripts/reproduce.py --config configs/examples/image.json --mode search --tag full
```

For only NeSCal or competitors, use `train_nescal.py` or `train_competitors.py` with the same arguments. `run_synthetic.py` runs the integrated synthetic suite.

Alternatively, pass chosen parameters explicitly:

```bash
python scripts/train_nescal.py --config configs/examples/image.json \
  --mode explicit --onset 60 --lambda-cosine 20 --tag chosen

python scripts/train_competitors.py --config configs/examples/image.json \
  --mode explicit --methods MMCE --method-params 'MMCE={"lambda_reg":7}' --tag mmce
```

These values illustrate the interface. Use `--lambda-constant VALUE` for constant regularization; synthetic NeSCal uses this option only. Competitor parameters and search ranges are defined in `src/configs/method_specs.py`.

## Common options

- `--dry-run`: inspect settings without training or downloading data.
- `--search-seeds 18 42 --confirmatory-seeds 613 2026`: set sweep and confirmation seeds.
- `--seeds 18 42 613 2026`: use these seeds without a separate confirmation phase.
- `--methods MMCE MDCA`: restrict competitor methods.
- `--set config.nescal.k=5`: override a configuration field.
- `--data-root PATH`, `--output PATH`, `--tag NAME`: set data and output locations.

## Evaluation and follow-up studies

Results are written to `runs/DATASET/ARCH/TAG`. Evaluate a trained run using the same configuration:

```bash
python scripts/evaluate_posthoc.py --config configs/examples/image.json \
  --run-root runs/cifar100/resnet50/full --calibrators ts local dc dac lore
```

Use `evaluate_corruptions.py`, `evaluate_ood.py` or `make_tables.py` with `--config` and `--run-root` for the other evaluations. The same interface applies to `run_ablations.py`, `run_random_graph.py` and `run_perturbations.py`. Each script's `--help` lists its options.

The additional experiments in the appendix start from completed image run roots and take one or more `--run-root` paths:

```bash
python scripts/run_ncr.py --run-root runs/cifar100/resnet50/full runs/cifar10/resnet50/full
python scripts/run_additional.py --run-root runs/cifar100/resnet50/full --stages factorial divergences
python scripts/run_backbones.py --stages head vit_ft
python scripts/additional_tables.py --run-root runs/cifar100/resnet50/full runs/cifar10/resnet50/full
```

`run_ncr.py` trains NCR under the NeSCal protocol on the coarse grid reported in the paper (`--grid full` adds the remaining cells). `run_additional.py` covers the per-setting analysis, the target and graph factorial, the alternative divergences, the scale and cache arms, and Mixup and RankMixup. `run_backbones.py` covers the frozen pre-trained backbones, ViT-B/16 fine-tuning and the cache benchmark. `additional_tables.py` prints the tables from their outputs.

The frozen-backbone heads read features extracted once with `python -m encoders.extract_embeddings --dataset imagenet --encoder dinov2` and `--dataset cifar100 --encoder vit_b16`.

## Paper results

| Result | Command |
|---|---|
| Main comparison, Brier score and accuracy, secondary calibration metrics, refinement, post-hoc composition, constant vs cosine schedule, training cost | `reproduce.py` for each configuration, then `make_tables.py` |
| CIFAR-100 onset check against a cosine learning rate | `reproduce.py --stages headline-selection` |
| Corruption robustness and SVHN detection | `evaluate_corruptions.py`, `evaluate_ood.py` |
| Graph controls and ablations | `run_random_graph.py`, `run_ablations.py` |
| Activation rules | `reproduce.py --stages heuristics` |
| Class imbalance and label noise | `run_perturbations.py` |
| Failure prediction and selection criterion | `analyze_results.py` |
| Across-setting Wilcoxon tests (from the reported setting means) | `paper_wilcoxon.py` |
| Theory diagnostics, and edge JS and edge agreement from activation onwards | `run_theory.py --kind real` |
| Synthetic study | `run_synthetic.py`, `run_theory.py --kind synthetic` |
| NCR | `run_ncr.py`, `additional_tables.py` |
| Label-fixed comparison, factorial, divergences, approximate graphs, graph rebuilding, target staleness, Mixup, RankMixup, paired comparisons | `run_additional.py`, `additional_tables.py` |
| Pre-trained backbones, ViT-B/16 fine-tuning, cost, memory and cache synchronization | `run_backbones.py`, `additional_tables.py` |

Experiment implementations are in `src/`; `scripts/` provides the launch commands. Optional orchestration checks run with `python -m unittest discover -s tests -v`; these do not run real training or verify the paper's numerical results.
