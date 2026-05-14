"""
Unified evaluation script supporting all three checkpoint architectures:
  - Baseline       (Backbone + mlp_3_layer)
  - B_Sig          (Backbone + mlp_3_layer_sigmoid_siglip)
  - B_Sig_Gate     (Backbone + MLP3_Gated  with learnable gating activations)

Architecture is auto-detected from the checkpoint's mlp.pt state-dict keys and
can be overridden via --arch.

The easiest way to evaluate is by named run from the built-in registry:

    python eval_checkpoint.py --list                    # discover available runs
    python eval_checkpoint.py --run Gating_CLIVE        # evaluate one
    python eval_checkpoint.py --run Sigmoid_KonIQ_to_CLIVE

For ad-hoc evaluation of user-trained checkpoints, use the escape-hatch flags:

    python eval_checkpoint.py \
        --checkpoint best_checkpoints/AGM_seed8_train_CLIVE_test_CLIVE \
        --dataset CLIVE

    python eval_checkpoint.py --checkpoint /path/to/ckpt --dataset CLIVE --arch baseline_sig

Author: Ankit Yadav
"""

import argparse
import json
import os
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from configs.default import MODEL_CONFIG, _make_dataset_paths
from dataset import (
    KonIQ_10K, CLIVE_inmemory, SPAQ, KADID10K, FLIVE, AGIQA3K, AGIQA1K,
)
from models import MLP3_Gated, mlp_3_layer, mlp_3_layer_sigmoid_siglip, SIGLIPWithMLP
from models.activations import ParamSigmoid2, ParamLeakyReLU2
from util import margin_loss, metric

warnings.simplefilter(action="ignore", category=FutureWarning)


# ╭──────────────────────────────────────────────────────────────────────╮
# │  Checkpoint registry                                                │
# ╰──────────────────────────────────────────────────────────────────────╯
#
# Single source of truth for shipped pretrained checkpoints. Used by both
# `eval_checkpoint.py --run <name>` and `eval_all.py`.
#
# Schema (per entry):
#   path        : checkpoint directory relative to repo root
#   arch        : one of ARCH_CHOICES; selects MLP head + activation handling
#   train       : dataset the checkpoint was trained on (informational)
#   test        : dataset the checkpoint was selected for (informational)
#   dataset_id  : default --dataset value to pass to the eval pipeline
#                 (e.g. "KonIQ_10K_CLIVE" for cross-dataset KonIQ -> CLIVE)
#   seed        : seed used for the train/val random_split at training time
#                 (only used when eval_split == "val_80_20")
#   eval_split  : "val_80_20" -> evaluate on the 20% held-out partition produced
#                                 by random_split(full_data, [0.8, 0.2], seed)
#                 "full"      -> evaluate on the full target dataset
#                                 (cross-dataset runs)

CHECKPOINT_REGISTRY = {
    "Gating_CLIVE": {
        "path": "pretrained_checkpoints/Baseline_param_activation_gating_MSE_seed8_train_CLIVE_test_CLIVE",
        "arch": "gating",
        "train": "CLIVE",
        "test": "CLIVE",
        "dataset_id": "CLIVE",
        "seed": 8,
        "eval_split": "val_80_20",
    },
    "Gating_KonIQ": {
        "path": "pretrained_checkpoints/Baseline_param_activation_gating_MSE_seed19_train_KonIQ_10K_test_KonIQ_10K",
        "arch": "gating",
        "train": "KonIQ_10K",
        "test": "KonIQ_10K",
        "dataset_id": "KonIQ_10K",
        "seed": 19,
        "eval_split": "val_80_20",
    },
    "Gating_KonIQ_to_CLIVE": {
        "path": "pretrained_checkpoints/Baseline_param_activation_gating_MSE_seed8_train_KonIQ_10K_test_CLIVE",
        "arch": "gating",
        "train": "KonIQ_10K",
        "test": "CLIVE",
        "dataset_id": "KonIQ_10K_CLIVE",
        "seed": 8,
        "eval_split": "full",
    },
    "Gating_CLIVE_to_KonIQ": {
        "path": "pretrained_checkpoints/Baseline_param_activation_gating_MSE_seed8_train_CLIVE_test_KonIQ_10K",
        "arch": "gating",
        "train": "CLIVE",
        "test": "KonIQ_10K",
        "dataset_id": "CLIVE_KonIQ_10K",
        "seed": 8,
        "eval_split": "full",
    },
    "Sigmoid_CLIVE": {
        "path": "pretrained_checkpoints/Baseline_Sigmoid_MSE_seed8_train_CLIVE_test_CLIVE",
        "arch": "baseline_sig",
        "train": "CLIVE",
        "test": "CLIVE",
        "dataset_id": "CLIVE",
        "seed": 8,
        "eval_split": "val_80_20",
    },
    "Sigmoid_KonIQ": {
        "path": "pretrained_checkpoints/Baseline_Sigmoid_MSE_seed8_train_KonIQ_10K_test_KonIQ_10K",
        "arch": "baseline_sig",
        "train": "KonIQ_10K",
        "test": "KonIQ_10K",
        "dataset_id": "KonIQ_10K",
        "seed": 8,
        "eval_split": "val_80_20",
    },
    "Sigmoid_KonIQ_to_CLIVE": {
        "path": "pretrained_checkpoints/Baseline_Sigmoid_MSE_seed8_train_KonIQ_10K_test_CLIVE",
        "arch": "baseline_sig",
        "train": "KonIQ_10K",
        "test": "CLIVE",
        "dataset_id": "KonIQ_10K_CLIVE",
        "seed": 8,
        "eval_split": "full",
    },
}

EVAL_SPLIT_CHOICES = ("full", "val_80_20")


def _eval_mode_str(entry: dict) -> str:
    """Compact human-readable label for a registry entry's eval mode."""
    if entry["eval_split"] == "val_80_20":
        return f"val_80_20 (s={entry['seed']})"
    return entry["eval_split"]


def _print_registry() -> None:
    """Pretty-print the checkpoint registry, then advise on usage."""
    header = ("Run name", "Arch", "Train", "Test", "Eval", "Path")
    rows = [
        (name, e["arch"], e["train"], e["test"], _eval_mode_str(e), e["path"])
        for name, e in CHECKPOINT_REGISTRY.items()
    ]
    widths = [
        max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(header)
    ]
    fmt = "  " + "  ".join(f"{{:<{w}}}" for w in widths)
    sep = "  " + "  ".join("-" * w for w in widths)

    print("\nAvailable checkpoint runs:\n")
    print(fmt.format(*header))
    print(sep)
    for r in rows:
        print(fmt.format(*r))
    print()
    print("Eval modes:")
    print("  val_80_20 (s=N) = 20% held-out partition produced by random_split with seed=N")
    print("                    (matches the training-time eval split)")
    print("  full            = entire target dataset (cross-dataset runs)")
    print()
    print("Usage:")
    print("  python eval_checkpoint.py --run <Run name>")
    print("  python eval_checkpoint.py --checkpoint <dir> --dataset <id> --seed <N>   "
          "(ad-hoc; default --eval_split is val_80_20)")
    print()


# ╭──────────────────────────────────────────────────────────────────────╮
# │  Architecture auto-detection                                        │
# ╰──────────────────────────────────────────────────────────────────────╯

ARCH_CHOICES = ("baseline", "baseline_sig", "gating")

# TODO Make it more robust
def detect_architecture(mlp_state_dict: dict, ckpt_dir: str) -> str:
    """Infer the MLP architecture from its state-dict key patterns and the
    checkpoint directory name.

    Returns one of ``ARCH_CHOICES``.
    """
    keys = set(mlp_state_dict.keys())

    has_gating_keys = any("act1.g" in k for k in keys)
    has_adapter_keys = any("adapter." in k for k in keys)

    if has_gating_keys:
        return "gating"

    if has_adapter_keys:
        name_lower = ckpt_dir.lower()
        if "sigmoid" in name_lower or "b_sig" in name_lower:
            return "baseline_sig"
        return "baseline"

    raise RuntimeError(
        f"Cannot auto-detect architecture from state-dict keys: "
        f"{sorted(keys)[:10]}...  Specify --arch explicitly."
    )


def build_mlp(arch: str, input_dim: int = 1152, hidden: int = 512,
              state_dict: dict | None = None) -> torch.nn.Module:
    """Instantiate the correct MLP head for the given architecture tag.
    """
    if arch == "baseline":
        return mlp_3_layer(input_dim=input_dim, hidden=hidden)
    elif arch == "baseline_sig":
        return mlp_3_layer_sigmoid_siglip(input_dim=input_dim, hidden=hidden)
    elif arch == "gating":
        use_gamma = False
        if state_dict is not None:
            use_gamma = any("sig_act.gamma" in k for k in state_dict)
        return MLP3_Gated(input_dim=input_dim, hidden=hidden,
                          use_gamma=use_gamma)
    else:
        raise ValueError(f"Unknown architecture '{arch}'. Choose from {ARCH_CHOICES}")


# ╭──────────────────────────────────────────────────────────────────────╮
# │  Dataset loader                                                     │
# ╰──────────────────────────────────────────────────────────────────────╯

def _load_eval_dataset(dataset_id: str, paths: dict,
                       seed: int = 8, eval_split: str = "full"):
    """Build the eval dataset, replicating the training-time split.

    For within-dataset runs (CLIVE/KonIQ/etc.), the training script does an
    80/20 random_split seeded by ``seed`` and uses the 20% partition as the
    eval set. Pass ``eval_split="val_80_20"`` to reproduce that protocol.

    For cross-dataset runs (KonIQ_10K_CLIVE etc.), the training script
    evaluates on the full target dataset; pass ``eval_split="full"``.
    """
    CROSS_DATASET_TEST = {
        "KonIQ_10K_CLIVE": "CLIVE",
        "CLIVE_KonIQ_10K": "KonIQ_10K",
    }
    resolved_id = CROSS_DATASET_TEST.get(dataset_id, dataset_id)

    constructors = {
        "CLIVE":     lambda: CLIVE_inmemory(path_to_db=paths["CLIVE"]),
        "KonIQ_10K": lambda: KonIQ_10K(path_to_db=paths["KonIQ_10K"]),
        "SPAQ":      lambda: SPAQ(path_to_db=paths["SPAQ"]),
        "KADID10K":  lambda: KADID10K(path_to_db=paths["KADID10K"]),
        "FLIVE":     lambda: FLIVE(path_to_db=paths["FLIVE"]),
        "AGIQA3K":   lambda: AGIQA3K(path_to_db=paths["AGIQA3K"]),
        "AGIQA1K":   lambda: AGIQA1K(path_to_db=paths["AGIQA1K"]),
    }
    if resolved_id not in constructors:
        raise ValueError(
            f"Unknown dataset '{dataset_id}' (resolved to '{resolved_id}'). "
            f"Choose from: {', '.join(constructors)} "
            f"or cross-dataset: {', '.join(CROSS_DATASET_TEST)}"
        )
    if resolved_id != dataset_id:
        print(f"Cross-dataset eval: '{dataset_id}' -> evaluating on '{resolved_id}'")

    full_data = constructors[resolved_id]()

    if eval_split == "full":
        print(f"Eval split: full dataset ({len(full_data)} samples)")
        return full_data
    elif eval_split == "val_80_20":
        total = len(full_data)
        train_len = int(0.8 * total)
        val_len = total - train_len
        _, val_data = random_split(
            full_data, [train_len, val_len],
            generator=torch.Generator().manual_seed(seed),
        )
        print(f"Eval split: val_80_20 with seed={seed} "
              f"-> {val_len}/{total} samples")
        return val_data
    else:
        raise ValueError(
            f"Unknown eval_split '{eval_split}'. "
            f"Choose from: {', '.join(EVAL_SPLIT_CHOICES)}"
        )


# ╭──────────────────────────────────────────────────────────────────────╮
# │  Main evaluation routine                                            │
# ╰──────────────────────────────────────────────────────────────────────╯

def run_eval(ckpt_dir: str, dataset_id: str, data_dir: str = "./Dataset",
             batch_size: int = 4, arch: str | None = None,
             device_str: str = "cuda",
             seed: int = 8, eval_split: str = "full"):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    dataset_paths = _make_dataset_paths(data_dir)

    # ── Load MLP state-dict & detect architecture ────────────────────────
    mlp_path = os.path.join(ckpt_dir, "mlp.pt")
    if not os.path.exists(mlp_path):
        raise FileNotFoundError(f"Expected MLP weights at {mlp_path}")

    raw_state = torch.load(mlp_path, map_location="cpu", weights_only=False)
    cleaned_state = {k.replace("module.", ""): v for k, v in raw_state.items()}

    if arch is None:
        arch = detect_architecture(cleaned_state, ckpt_dir)
    print(f"Architecture: {arch}")

    # ── Build MLP and load weights ───────────────────────────────────────
    mlp = build_mlp(arch, input_dim=MODEL_CONFIG["mlp_input_dim"],
                    state_dict=cleaned_state)
    mlp.load_state_dict(cleaned_state)
    mlp = mlp.to(device).to(torch.bfloat16)

    if arch == "gating":
        with torch.no_grad():
            for m in mlp.modules():
                if isinstance(m, (ParamSigmoid2, ParamLeakyReLU2)):
                    m.float()

    mlp.eval()

    # ── Load backbone (PEFT adapter or full weights) ─────────────────────
    adapter_cfg_path = os.path.join(ckpt_dir, "adapter_config.json")
    if os.path.exists(adapter_cfg_path):
        from peft import PeftModel, PeftConfig
        peft_cfg = PeftConfig.from_pretrained(ckpt_dir)
        base_model_id = peft_cfg.base_model_name_or_path
        print(f"Loading base model: {base_model_id}")
        model = AutoModel.from_pretrained(
            base_model_id, torch_dtype=torch.bfloat16
        ).to(device)
        print(f"Loading LoRA adapter from: {ckpt_dir}")
        model = PeftModel.from_pretrained(model, ckpt_dir).to(device)
        model = model.merge_and_unload()
    else:
        print(f"Loading full model from: {ckpt_dir}")
        model = AutoModel.from_pretrained(
            ckpt_dir, torch_dtype=torch.bfloat16
        ).to(device)

    model.eval()

    processor = AutoProcessor.from_pretrained(
        MODEL_CONFIG["model_id"], local_files_only=True
    )

    # ── Dataset ──────────────────────────────────────────────────────────
    dataset = _load_eval_dataset(dataset_id, dataset_paths,
                                 seed=seed, eval_split=eval_split)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         drop_last=False, num_workers=4, pin_memory=True)

    # ── Wrap for unified forward ─────────────────────────────────────────
    combined = SIGLIPWithMLP(
        base_model=model.float(),
        mlp_head=mlp.float(),
        device=device,
    ).to(device).eval()

    # ── Inference loop ───────────────────────────────────────────────────
    all_preds, all_labels = [], []
    total_loss, n_batches = 0.0, 0

    with torch.no_grad(), \
         torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        for batch in tqdm(loader, desc=f"Evaluating on {dataset_id}"):
            images = batch["image"].to(device)
            gt     = batch["score"].to(device)
            inputs = processor(images=images, return_tensors="pt").to(device)
            preds  = combined(inputs["pixel_values"])

            loss = (
                torch.nn.functional.mse_loss(preds, gt)
                + margin_loss(preds, gt)
            )
            total_loss += loss.item()
            n_batches  += 1

            all_preds.append(preds.float().cpu().numpy())
            all_labels.append(gt.float().cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    m = metric()
    m.calcuate_srcc(y_true, y_pred)
    m.calculate_plcc(y_true, y_pred)
    avg_loss = total_loss / max(n_batches, 1)

    results = {
        "dataset":       dataset_id,
        "checkpoint":    ckpt_dir,
        "architecture":  arch,
        "eval_split":    eval_split,
        "seed":          seed if eval_split == "val_80_20" else None,
        "SRCC":          float(m.result["SRCC"]),
        "PLCC":          float(m.result["PLCC"]),
        "avg_eval_loss": float(avg_loss),
        "num_samples":   int(len(y_pred)),
    }

    print("\n========== Evaluation Results ==========")
    for k, v in results.items():
        print(f"  {k:16s}: {v}")
    print("========================================\n")

    return results


# ╭──────────────────────────────────────────────────────────────────────╮
# │  CLI entry point                                                    │
# ╰──────────────────────────────────────────────────────────────────────╯

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint on a dataset (supports Baseline, B_Sig, and B_Sig_Gate)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all available named runs from the built-in registry and exit.",
    )
    parser.add_argument(
        "--run", type=str, default=None,
        help="Name of a registered checkpoint run (see --list).\n"
             "Resolves --checkpoint, --arch, and --dataset automatically.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a checkpoint directory (must contain mlp.pt).\n"
             "Escape hatch for user-trained checkpoints not in the registry.",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Dataset to evaluate on. Required when --run is not provided.\n"
             "Choices: CLIVE, KonIQ_10K, SPAQ, KADID10K, FLIVE, AGIQA3K, AGIQA1K,\n"
             "         KonIQ_10K_CLIVE, CLIVE_KonIQ_10K\n"
             "Overrides the registry's default when used with --run.",
    )
    parser.add_argument(
        "--arch", type=str, default=None, choices=ARCH_CHOICES,
        help="Override auto-detected architecture.\n"
             "  baseline     = Backbone + mlp_3_layer (ReLU)\n"
             "  baseline_sig = Backbone + mlp_3_layer_sigmoid_siglip (Sigmoid+LeakyReLU)\n"
             "  gating       = Backbone + MLP3_Gated (learnable gated activations)",
    )
    parser.add_argument(
        "--eval_split", type=str, default=None, choices=EVAL_SPLIT_CHOICES,
        help="Which subset of the dataset to evaluate on.\n"
             "  val_80_20 = 20%% held-out partition (requires --seed)\n"
             "  full      = the entire target dataset (cross-dataset evals)\n"
             "If --run is set, defaults to the registry's value; otherwise val_80_20.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Seed for the train/val random_split when --eval_split=val_80_20.\n"
             "If --run is set, defaults to the registry's value;\n"
             "otherwise required when --eval_split=val_80_20.",
    )
    parser.add_argument(
        "--data_dir", type=str, default="./Dataset",
        help="Root directory containing dataset subfolders.",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device string, e.g. cuda, cuda:0, cuda:1, cpu")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to this JSON file.\n"
                             "Default: results/eval_checkpoint_<run-or-dataset>.json")

    args = parser.parse_args()

    if args.list:
        _print_registry()
        return

    # ── Resolve checkpoint / arch / dataset / split / seed
    #     from --run or escape-hatch flags
    run_name = None
    if args.run is not None:
        if args.run not in CHECKPOINT_REGISTRY:
            parser.error(
                f"Unknown run '{args.run}'. Use --list to see available runs."
            )
        entry = CHECKPOINT_REGISTRY[args.run]
        run_name = args.run
        ckpt_dir   = args.checkpoint or entry["path"]
        arch       = args.arch       or entry["arch"]
        dataset_id = args.dataset    or entry["dataset_id"]
        eval_split = args.eval_split or entry["eval_split"]
        seed       = args.seed       if args.seed is not None else entry["seed"]
    else:
        if args.checkpoint is None or args.dataset is None:
            parser.error(
                "Specify either --run <name> (see --list) or both "
                "--checkpoint <dir> and --dataset <id>."
            )
        ckpt_dir   = args.checkpoint
        arch       = args.arch
        dataset_id = args.dataset
        eval_split = args.eval_split or "val_80_20"
        if eval_split == "val_80_20" and args.seed is None:
            parser.error(
                "--seed is required when --eval_split=val_80_20 "
                "(use the seed that was used for the training-time random_split, "
                "or pass --eval_split=full to evaluate on the entire dataset)."
            )
        seed = args.seed if args.seed is not None else 8

    if not os.path.isdir(ckpt_dir):
        parser.error(f"Checkpoint directory does not exist: {ckpt_dir}")

    results = run_eval(
        ckpt_dir=ckpt_dir,
        dataset_id=dataset_id,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        arch=arch,
        device_str=args.device,
        seed=seed,
        eval_split=eval_split,
    )
    if run_name is not None:
        results["run"] = run_name

    # Save results
    out_path = args.output
    if out_path is None:
        os.makedirs("results", exist_ok=True)
        tag = run_name if run_name is not None else dataset_id
        out_path = f"results/eval_checkpoint_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
