"""
Standalone evaluation script for NR-IQA with Activation-Gated MLP (AGM).

Loads a saved checkpoint (backbone + MLP weights) and evaluates on a chosen
dataset, reporting SRCC, PLCC, and average loss.  Optionally generates
GradCAM visualisations for the first batch of images.

Usage examples
--------------
    # Auto-detect best checkpoint for CLIVE
    python eval.py --dataset CLIVE

    # Evaluate a specific checkpoint directory
    python eval.py --dataset KonIQ_10K --checkpoint_dir best_checkpoints/AGM_seed8_train_KonIQ_10K_test_KonIQ_10K

    # Evaluate on full dataset (no GradCAM)
    python eval.py --dataset SPAQ --no_gradcam

    # Save results to a custom path
    python eval.py --dataset AGIQA3K --output results/agiqa3k_eval.json

Author: Ankit Yadav
"""

import argparse
import json
import os
import warnings

import cv2
import numpy as np
import torch
from glob import glob
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from configs.default import MODEL_CONFIG, DATASET_PATHS, _make_dataset_paths
from dataset import (
    KonIQ_10K, CLIVE_inmemory, SPAQ, KADID10K, FLIVE, AGIQA3K, AGIQA1K,
)
from models import MLP3_Gated, SIGLIPWithMLP
from models.activations import ParamSigmoid2, ParamLeakyReLU2
from util import margin_loss, metric, Overlay

warnings.simplefilter(action="ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Dataset builder (eval-only: returns the full dataset, no split)
# ---------------------------------------------------------------------------

def _load_eval_dataset(dataset_id: str, paths: dict):
    """For cross-dataset ids like ``KonIQ_10K_CLIVE`` the *test* set is the
    second part (CLIVE), matching the convention used during training."""
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
        print(f"Cross-dataset eval: '{dataset_id}' — evaluating on test set '{resolved_id}'")
    return constructors[resolved_id]()


# ---------------------------------------------------------------------------
# Checkpoint auto-detection
# ---------------------------------------------------------------------------

PRETRAINED_CHECKPOINTS = {
    "CLIVE": "pretrained_checkpoints/Baseline_param_activation_gating_MSE_seed8_step_train_CLIVE_TestCLIVE_14010",
}


def _find_checkpoint(stage_name: str, dataset_id: str) -> str:
    """Search for a checkpoint in this order:
    1. ``pretrained_checkpoints/`` (shipped with the repo)
    2. ``best_checkpoints/`` (produced by training)
    Falls back to a broader glob if the stage-name prefix doesn't match.
    """
    # 1) Pretrained checkpoints shipped with the repo
    if dataset_id in PRETRAINED_CHECKPOINTS:
        path = PRETRAINED_CHECKPOINTS[dataset_id]
        if os.path.isdir(path):
            print(f"Using pretrained checkpoint: {path}")
            return path

    # 2) User-trained best checkpoints
    for search_dir in ("best_checkpoints", "pretrained_checkpoints"):
        pattern = f"{search_dir}/{stage_name}*_test_{dataset_id}"
        matches = sorted(glob(pattern), key=os.path.getctime)
        if not matches:
            pattern = f"{search_dir}/*_test_{dataset_id}"
            matches = sorted(glob(pattern), key=os.path.getctime)
        if matches:
            chosen = matches[-1]
            print(f"Auto-selected checkpoint: {chosen}")
            return chosen

    raise FileNotFoundError(
        f"No checkpoint found for dataset '{dataset_id}' under "
        "pretrained_checkpoints/ or best_checkpoints/. "
        "Please specify --checkpoint_dir explicitly."
    )


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_paths = _make_dataset_paths(args.data_dir)

    # ── Locate checkpoint ────────────────────────────────────────────────
    ckpt_dir = args.checkpoint_dir
    if ckpt_dir is None:
        ckpt_dir = _find_checkpoint(args.stage_name, args.dataset)

    # ── Load backbone ────────────────────────────────────────────────────
    print(f"Loading backbone from {ckpt_dir} ...")
    model = AutoModel.from_pretrained(ckpt_dir, torch_dtype=torch.bfloat16).to(device)
    processor = AutoProcessor.from_pretrained(args.model_id)

    # ── Load MLP head ────────────────────────────────────────────────────
    mlp = MLP3_Gated(input_dim=args.mlp_input_dim).to(device).to(torch.bfloat16)
    with torch.no_grad():
        for m in mlp.modules():
            if isinstance(m, (ParamSigmoid2, ParamLeakyReLU2)):
                m.float()

    mlp_path = os.path.join(ckpt_dir, "mlp.pt")
    if not os.path.exists(mlp_path):
        raise FileNotFoundError(f"Expected MLP weights at {mlp_path}")

    state = torch.load(mlp_path, map_location=device)
    try:
        mlp.load_state_dict(state)
    except RuntimeError:
        cleaned = {k.replace("module.", ""): v for k, v in state.items()}
        mlp.load_state_dict(cleaned)

    model.eval()
    mlp.eval()

    # ── Dataset ──────────────────────────────────────────────────────────
    dataset = _load_eval_dataset(args.dataset, dataset_paths)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # ── Wrap for unified forward ─────────────────────────────────────────
    combined = SIGLIPWithMLP(
        base_model=model.float(),
        mlp_head=mlp.float(),
        device=device,
    ).to(device).eval()

    # ── Inference loop ───────────────────────────────────────────────────
    all_preds, all_labels = [], []
    total_loss, n_batches = 0.0, 0

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        for batch in tqdm(loader, desc=f"Evaluating on {args.dataset}"):
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

            all_preds.append(preds.float().detach().cpu().numpy())
            all_labels.append(gt.float().detach().cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    m = metric()
    srcc = m.calcuate_srcc(y_true, y_pred)
    plcc = m.calculate_plcc(y_true, y_pred)
    avg_loss = total_loss / max(n_batches, 1)

    results = {
        "dataset":       args.dataset,
        "checkpoint":    ckpt_dir,
        "SRCC":          float(srcc),
        "PLCC":          float(plcc),
        "avg_eval_loss": float(avg_loss),
        "num_samples":   len(y_pred),
    }

    print("\n========== Evaluation Results ==========")
    for k, v in results.items():
        print(f"  {k:16s}: {v}")
    print("========================================\n")

    # ── Save results ─────────────────────────────────────────────────────
    out_path = args.output
    if out_path is None:
        os.makedirs("results", exist_ok=True)
        out_path = f"results/eval_{args.stage_name}_{args.dataset}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results written to {out_path}")

    # ── GradCAM (optional) ───────────────────────────────────────────────
    if not args.no_gradcam:
        try:
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.model_targets import RawScoresOutputTarget

            os.makedirs("grad_cam_eval", exist_ok=True)

            target_layers = [
                combined.siglip.base_model.vision_model.embeddings.patch_embedding
            ]
            cam = GradCAM(model=combined, target_layers=target_layers)

            first_batch = next(iter(loader))
            imgs = first_batch["image"].to(device)
            inp  = processor(images=imgs, return_tensors="pt").to(device)
            preds_cam = combined(inp["pixel_values"])
            targets   = [RawScoresOutputTarget() for _ in range(preds_cam.size(0))]

            for p in target_layers[0].parameters():
                p.requires_grad = True

            with torch.amp.autocast(device_type="cuda", enabled=False):
                cam.model = cam.model.float()
                grayscale_cams = cam(
                    input_tensor=inp["pixel_values"].to(torch.float),
                    targets=targets,
                )

            saved = []
            for idx, (img_t, hm) in enumerate(zip(imgs.cpu(), grayscale_cams)):
                vis, _ = Overlay(img_t, hm, alpha=(0.8, 0.4))
                path = f"grad_cam_eval/{args.dataset}_cam_{idx}.jpg"
                cv2.imwrite(path, vis)
                saved.append(path)
            print(f"GradCAM visualisations saved: {saved}")
        except Exception as e:
            print(f"GradCAM generation failed: {e}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate NR-IQA AGM checkpoint")

    p.add_argument("--dataset", type=str, required=True,
                   help="CLIVE, KonIQ_10K, SPAQ, KADID10K, FLIVE, AGIQA3K, AGIQA1K")
    p.add_argument("--data_dir", type=str, default="./Dataset")
    p.add_argument("--checkpoint_dir", type=str, default=None,
                   help="Path to checkpoint directory (auto-detected if omitted)")
    p.add_argument("--stage_name", type=str, default="AGM_seed8")
    p.add_argument("--model_id", type=str, default=MODEL_CONFIG["model_id"])
    p.add_argument("--mlp_input_dim", type=int, default=MODEL_CONFIG["mlp_input_dim"])
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--output", type=str, default=None,
                   help="Path to save results JSON (default: results/eval_<stage>_<dataset>.json)")
    p.add_argument("--no_gradcam", action="store_true",
                   help="Skip GradCAM visualisation")

    return p.parse_args()


if __name__ == "__main__":
    run_eval(parse_args())
