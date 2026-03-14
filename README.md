# NR-IQA AGM: No-Reference Image Quality Assessment with Activation-Gated MLP

A SigLIP-2 based no-reference image quality assessment model that uses **learnable activation gating** (a blend of parametric sigmoid and parametric leaky-ReLU) in its quality-prediction MLP head.

## Architecture

```
Image -> SigLIP-2 Vision Encoder (+ LoRA / DPT) -> [CLS] features (1152-d)
                                                          |
                                                    MLP3_Gated
                                                    ├── Linear(1152, 512)
                                                    ├── GatedBlend(ParamSigmoid + ParamLeakyReLU)
                                                    ├── Linear(512, 512)
                                                    ├── ParamLeakyReLU
                                                    └── Linear(512, 1) -> quality score
```

**Loss**: MSE + pair-wise margin ranking loss

## Project Structure

```
NR_IQA_AGM/
├── configs/
│   ├── __init__.py
│   └── default.py            # Default model / training / dataset configs
├── models/
│   ├── __init__.py
│   ├── activations.py        # ParamSigmoid2, ParamLeakyReLU2, GatedBlend
│   ├── mlp_heads.py          # MLP3_Gated, mlp_3_layer
│   └── wrappers.py           # SIGLIPWithMLP (GradCAM-compatible wrapper)
├── Dataset/                  # <-- Place your datasets here (see below)
├── pretrained_checkpoints/   # Pretrained weights (CLIVE, CLIVE->KonIQ, KonIQ->CLIVE)
├── checkpoints/              # Training checkpoints (auto-created)
├── best_checkpoints/         # Best model per dataset (auto-created)
├── resume_state/             # Resume state files (auto-created)
├── results/                  # Evaluation JSON results (auto-created)
├── dataset.py                # PyTorch Dataset classes for all IQA benchmarks
├── seed.py                   # Reproducibility (seed = 8)
├── util.py                   # Clean utilities (loss, metrics, Overlay, constants)
├── util_main.py              # Original util.py (archived, contains all experiments)
├── train.py                  # Training entry point
├── eval.py                   # Evaluation entry point
├── requirements.txt          # pip dependencies
├── environment.yml           # Conda environment spec
└── README.md
```

## Dataset Setup

Create a `Dataset/` folder in the project root and organise each benchmark as shown below.
The exact sub-directory names and annotation files **must** match what `dataset.py` expects.

```
Dataset/
├── KonIQ_10K/
│   ├── koniq10k_512x384/
│   │   └── 512x384/                # 10,073 images
│   └── koniq10k_scores_and_distributions/
│       └── koniq10k_scores_and_distributions.csv
│
├── CLIVE/
│   └── ChallengeDB_release/
│       ├── Data/
│       │   ├── AllImages_release.mat
│       │   ├── AllMOS_release.mat
│       │   └── AllStdDev_release.mat
│       └── Images/                  # 1,162 images
│
├── SPAQ/
│   ├── SPAQ_dataset/
│   │   └── Annotations/
│   │       └── MOS_and_Image_attribute_scores.xlsx
│   └── TestImage/                   # 11,125 images
│
├── KADID-10K/
│   └── kadid10k/
│       ├── dmos.csv
│       └── images/                  # 10,125 images
│
├── FLIVE/
│   ├── labels_image.csv
│   └── database/                    # ~40,000 images (sub-folders inside)
│
├── AGIQA-3k/
│   ├── data.csv
│   └── images/                      # 2,982 images
│
└── AGIQA-1k/
    ├── AIGC_MOS_Zscore.xlsx
    └── images/                      # 1,000 images
```

> **Tip**: You can symlink existing dataset directories instead of copying:
> ```bash
> ln -s /path/to/your/KonIQ_10K Dataset/KonIQ_10K
> ```

## Installation

### Option A: pip (inside an existing environment)

```bash
pip install -r requirements.txt
```

> **Note**: Install PyTorch with the CUDA version matching your GPU driver
> first. See [pytorch.org/get-started](https://pytorch.org/get-started/locally/).

### Option B: Conda (creates a fresh environment)

```bash
conda env create -f environment.yml
conda activate nr_iqa_agm
```

> Edit `pytorch-cuda=12.1` in `environment.yml` if you need a different
> CUDA version (e.g. `11.8`).

## Training

```bash
# Train on KonIQ-10K with default hyperparameters
python train.py --dataset KonIQ_10K

# Train on CLIVE with LoRA rank 8, 20 epochs, batch size 4
python train.py --dataset CLIVE --peft_method LoRA --lora_r 8 --epochs 20 --batch_size 4

# Cross-dataset: train on KonIQ-10K, evaluate on CLIVE
python train.py --dataset KonIQ_10K_CLIVE

# Full fine-tuning (no PEFT adapter)
python train.py --dataset SPAQ --peft_method NA

# Deep Prompt Tuning instead of LoRA
python train.py --dataset KADID10K --peft_method DPT

# Resume a previous run
python train.py --dataset KonIQ_10K --resume

# Dry-run for quick debugging (100 train batches, 32 eval batches)
python train.py --dataset CLIVE --dry_run

# Disable WandB logging
python train.py --dataset CLIVE --no_wandb
```

### Key Training Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | *required* | Dataset to train on (see list above) |
| `--data_dir` | `./Dataset` | Root directory of all datasets |
| `--model_id` | `google/siglip2-so400m-patch16-512` | HuggingFace backbone |
| `--peft_method` | `LoRA` | `LoRA`, `DPT`, or `NA` |
| `--epochs` | `15` | Number of training epochs |
| `--batch_size` | `2` | Per-device batch size |
| `--lr` | `1e-4` | Learning rate |
| `--grad_accum` | `6` | Gradient accumulation steps (effective batch = batch_size * grad_accum) |
| `--lr_milestones` | `30,35` | Comma-separated epoch milestones for MultiStepLR |
| `--checkpoint_steps` | `5000` | Save a checkpoint every N steps |
| `--stage_name` | `AGM_seed8` | Prefix for checkpoint directories |
| `--resume` | off | Resume from the latest `resume_state/` file |
| `--dry_run` | off | Fast debugging mode |
| `--no_wandb` | off | Disable Weights & Biases logging |
| `--no_eval` | off | Skip evaluation during training |
| `--eval_every` | `1` | Evaluate every N epochs |

## Pretrained Checkpoints

The repo ships with pretrained weights in `pretrained_checkpoints/`.
These use the **MLP3_Gated** head (activation gating) with LoRA on SigLIP-2:

| Train set | Test set | Checkpoint directory |
|-----------|----------|---------------------|
| CLIVE | CLIVE | `pretrained_checkpoints/Baseline_param_activation_gating_MSE_seed8_step_train_CLIVE_TestCLIVE_14010` |

> More pretrained checkpoints (cross-dataset) will be added in subsequent updates.

`eval.py` automatically picks the pretrained checkpoint when available — no `--checkpoint_dir` needed.

## Evaluation

```bash
# Evaluate using the pretrained CLIVE->CLIVE checkpoint (auto-detected)
python eval.py --dataset CLIVE

# Cross-dataset: pretrained KonIQ_10K->CLIVE checkpoint
python eval.py --dataset KonIQ_10K_CLIVE

# Evaluate a specific (user-trained) checkpoint
python eval.py --dataset KonIQ_10K \
    --checkpoint_dir best_checkpoints/AGM_seed8_train_KonIQ_10K_test_KonIQ_10K

# Skip GradCAM visualisation
python eval.py --dataset SPAQ --no_gradcam

# Custom output path
python eval.py --dataset AGIQA3K --output my_results.json
```

### Key Evaluation Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | *required* | Dataset to evaluate on |
| `--data_dir` | `./Dataset` | Root directory of all datasets |
| `--checkpoint_dir` | auto | Path to checkpoint dir (auto-detected from `best_checkpoints/`) |
| `--batch_size` | `4` | Evaluation batch size |
| `--no_gradcam` | off | Skip GradCAM heatmap generation |
| `--output` | auto | Path to save results JSON |

## Evaluation Metrics

- **SRCC** (Spearman Rank-order Correlation Coefficient): measures monotonic association between predicted and ground-truth scores.
- **PLCC** (Pearson Linear Correlation Coefficient): measures linear correlation after fitting.

## Supported Datasets

| Dataset | Type | # Images | Score Range |
|---------|------|----------|-------------|
| KonIQ-10K | Authentic distortions | 10,073 | MOS / 100 |
| CLIVE | Authentic distortions | 1,162 | MOS / 100 |
| SPAQ | Smartphone photos | 11,125 | MOS / 100 |
| KADID-10K | Synthetic distortions | 10,125 | (DMOS - 1) / 4 |
| FLIVE | Authentic (in-the-wild) | ~40,000 | MOS / 100 |
| AGIQA-3K | AI-generated | 2,982 | MOS_quality / 5 |
| AGIQA-1K | AI-generated | 1,000 | MOS / 5 |

## Cross-Dataset Experiments

Pass a combined dataset ID to `train.py`:

- `KonIQ_10K_CLIVE` — train on KonIQ-10K, evaluate on CLIVE
- `CLIVE_KonIQ_10K` — train on CLIVE, evaluate on KonIQ-10K

## License

See [LICENSE](LICENSE).

## TODO

- [ ] Add pretrained checkpoints for KonIQ-10K, SPAQ, KADID-10K, FLIVE, AGIQA-3K, AGIQA-1K
- [ ] Add cross-dataset pretrained checkpoints (KonIQ-10K -> CLIVE, CLIVE -> KonIQ-10K, etc.)

## Citation

If you use this codebase in your research, please cite:

```bibtex
@InProceedings{Yadav_2026_WACV,
    author    = {Yadav, Ankit and Huy, Ta Duc and Liu, Lingqiao},
    title     = {Revisiting Vision-Language Foundations for No-Reference Image Quality Assessment},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {March},
    year      = {2026},
    pages     = {5416-5425}
}
```
