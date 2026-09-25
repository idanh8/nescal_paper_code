# NeSCal

Code for reproducing the NeSCal experiments from the paper *Neighborhood Smoothing for Calibration*. NeSCal regularizes predictive disagreement between neighbors in embedding space, using a cosine kNN graph fixed after warm-up.

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

## Basic options examples

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

Experiment implementations are in `src/`; `scripts/` provides the launch commands. Optional orchestration checks run with `python -m unittest discover -s tests -v`; these do not run real training or verify the paper's numerical results.
