"""
Training script for NR-IQA with Activation-Gated MLP (AGM).

Backbone : SigLIP-2 vision encoder  (PEFT: LoRA or Deep Prompt Tuning)
Head     : MLP3_Gated (learnable activation gating)
Loss     : MSE + pair-wise margin loss

Usage examples
--------------
    # Train on KonIQ-10K with default settings
    python train.py --dataset KonIQ_10K

    # Train on CLIVE, LoRA rank 8, 20 epochs
    python train.py --dataset CLIVE --peft_method LoRA --lora_r 8 --epochs 20

    # Cross-dataset: train on KonIQ-10K, evaluate on CLIVE
    python train.py --dataset KonIQ_10K_CLIVE

    # Resume from a previous run
    python train.py --dataset KonIQ_10K --resume

    # Dry-run (100 train steps per epoch) for debugging
    python train.py --dataset CLIVE --dry_run

Author: Ankit Yadav
"""

import argparse
import copy
import json
import os
import warnings

import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator
from glob import glob
from peft import (
    LoraConfig,
    PromptEncoderConfig,
    TaskType,
    get_peft_model,
)
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

import wandb

from configs.default import MODEL_CONFIG, TRAIN_CONFIG, DATASET_PATHS, _make_dataset_paths
from dataset import (
    KonIQ_10K, KonIQ_10K_inmemory,
    CLIVE, CLIVE_inmemory,
    SPAQ, KADID10K, FLIVE,
    AGIQA3K, AGIQA1K,
)
from models import MLP3_Gated, SIGLIPWithMLP
from models.activations import ParamSigmoid2, ParamLeakyReLU2
from seed import Seed, seed_worker
from util import margin_loss, metric, Overlay, BAD_QUALITY_PROMPT, Text_Template_baseline

warnings.simplefilter(action="ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_datasets(dataset_id: str, paths: dict, seed: int):
    """Return ``(train_dataset, eval_dataset)`` for the requested *dataset_id*.

    Supports intra-dataset 80/20 splits as well as cross-dataset pairs
    like ``KonIQ_10K_CLIVE`` (train on KonIQ, test on CLIVE).
    """
    gen = torch.Generator().manual_seed(seed)

    if dataset_id == "CLIVE":
        full = CLIVE_inmemory(path_to_db=paths["CLIVE"])
        t_len = int(0.8 * len(full))
        return random_split(full, [t_len, len(full) - t_len], generator=gen)

    if dataset_id == "KonIQ_10K":
        full = KonIQ_10K(path_to_db=paths["KonIQ_10K"])
        t_len = int(0.8 * len(full))
        return random_split(full, [t_len, len(full) - t_len], generator=gen)

    if dataset_id == "KonIQ_10K_CLIVE":
        train_ds = KonIQ_10K(path_to_db=paths["KonIQ_10K"])
        eval_ds  = CLIVE_inmemory(path_to_db=paths["CLIVE"])
        train_ds, _ = random_split(train_ds, [len(train_ds), 0], generator=gen)
        eval_ds,  _ = random_split(eval_ds,  [len(eval_ds),  0], generator=gen)
        return train_ds, eval_ds

    if dataset_id == "CLIVE_KonIQ_10K":
        train_ds = CLIVE_inmemory(path_to_db=paths["CLIVE"])
        eval_ds  = KonIQ_10K(path_to_db=paths["KonIQ_10K"])
        train_ds, _ = random_split(train_ds, [len(train_ds), 0], generator=gen)
        eval_ds,  _ = random_split(eval_ds,  [len(eval_ds),  0], generator=gen)
        return train_ds, eval_ds

    if dataset_id == "SPAQ":
        full = SPAQ(path_to_db=paths["SPAQ"])
        t_len = int(0.8 * len(full))
        return random_split(full, [t_len, len(full) - t_len], generator=gen)

    if dataset_id == "KADID10K":
        full = KADID10K(path_to_db=paths["KADID10K"])
        t_len = int(0.8 * len(full))
        return random_split(full, [t_len, len(full) - t_len], generator=gen)

    if dataset_id == "FLIVE":
        full = FLIVE(path_to_db=paths["FLIVE"])
        t_len = int(0.8 * len(full))
        return random_split(full, [t_len, len(full) - t_len], generator=gen)

    if dataset_id == "AGIQA3K":
        full = AGIQA3K(path_to_db=paths["AGIQA3K"])
        t_len = int(0.8 * len(full))
        return random_split(full, [t_len, len(full) - t_len], generator=gen)

    if dataset_id == "AGIQA1K":
        full = AGIQA1K(path_to_db=paths["AGIQA1K"])
        t_len = int(0.8 * len(full))
        return random_split(full, [t_len, len(full) - t_len], generator=gen)

    raise ValueError(
        f"Unknown dataset_id '{dataset_id}'. "
        "Choose from: CLIVE, KonIQ_10K, SPAQ, KADID10K, FLIVE, "
        "AGIQA3K, AGIQA1K, KonIQ_10K_CLIVE, CLIVE_KonIQ_10K"
    )


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _clean_old_checkpoints(stage_name: str, max_keep: int):
    for pattern in (f"checkpoints/{stage_name}_step*.pt",
                    f"checkpoints/{stage_name}_step*/"):
        items = sorted(glob(pattern), key=os.path.getctime)
        if len(items) > max_keep:
            for p in items[:-max_keep]:
                os.system(f"rm -rf {p}")


def _db_name(loader):
    """Walk through Subset wrappers to find the underlying ``db_name``."""
    ds = loader.dataset
    while hasattr(ds, "dataset"):
        ds = ds.dataset
    return getattr(ds, "db_name", "unknown")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, mlp, processor, dataloader_eval, device, dry_run=False):
    """Run inference on *dataloader_eval* and return (results_dict, avg_loss).

    The model and MLP are deep-copied so evaluation doesn't affect the
    training-mode state.
    """
    model_copy = copy.deepcopy(model)
    mlp_copy   = copy.deepcopy(mlp)

    base = model_copy.module if hasattr(model_copy, "module") else model_copy
    combined = SIGLIPWithMLP(
        base_model=base.float(),
        mlp_head=mlp_copy.float(),
        device=device,
    ).to(device).eval()

    all_preds, all_labels = [], []
    total_loss, n_batches = 0.0, 0
    cnt = 0

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        for batch in tqdm(dataloader_eval, desc="Evaluating"):
            if dry_run:
                cnt += 1
                if cnt >= 32:
                    break

            images   = batch["image"].to(device)
            gt       = batch["score"].to(device)
            inputs   = processor(images=images, return_tensors="pt").to(device)
            preds    = combined(inputs["pixel_values"])

            loss_mse = torch.nn.functional.mse_loss(preds, gt)
            loss_mrg = margin_loss(preds, gt)
            total_loss += (loss_mse + loss_mrg).item()
            n_batches  += 1

            all_preds.append(preds.float().cpu().numpy())
            all_labels.append(gt.float().cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    m = metric()
    m.calcuate_srcc(y_true, y_pred)
    m.calculate_plcc(y_true, y_pred)

    avg_loss = total_loss / max(n_batches, 1)
    results  = dict(m.result)
    results["avg_eval_loss"] = float(avg_loss)

    del combined, model_copy, mlp_copy
    torch.cuda.empty_cache()
    return results, avg_loss


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    cfg = {**TRAIN_CONFIG}
    cfg.update({
        "epochs":                       args.epochs,
        "batch_size":                   args.batch_size,
        "learning_rate":                args.lr,
        "weight_decay":                 args.weight_decay,
        "checkpoint_steps":             args.checkpoint_steps,
        "max_checkpoints":              args.max_checkpoints,
        "stage_name":                   args.stage_name,
        "peft_method":                  args.peft_method,
        "gradient_accumulation_steps":  args.grad_accum,
        "do_eval":                      not args.no_eval,
        "eval_epoch_steps":             args.eval_every,
        "lr_scheduler":                 not args.no_scheduler,
        "lr_scheduler_milestones":      list(map(int, args.lr_milestones.split(","))),
        "use_gradient_clip":            args.gradient_clip > 0,
        "gradient_clip":                args.gradient_clip,
        "resume":                       args.resume,
        "dry_run":                      args.dry_run,
        "wandb_project":                args.wandb_project,
    })
    if args.lora_r is not None:
        cfg["lora_config"]["r"]            = args.lora_r
        cfg["lora_config"]["lora_alpha"]   = args.lora_alpha
        cfg["lora_config"]["lora_dropout"] = args.lora_dropout

    dataset_paths = _make_dataset_paths(args.data_dir)

    accelerator = Accelerator(mixed_precision="no")
    device = accelerator.device

    if dist.is_initialized():
        dist.barrier(device_ids=[torch.cuda.current_device()])

    # ── Processor ────────────────────────────────────────────────────────
    processor = AutoProcessor.from_pretrained(args.model_id)

    # ── Backbone ─────────────────────────────────────────────────────────
    model = AutoModel.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to(device)

    # ── PEFT ─────────────────────────────────────────────────────────────
    if cfg["peft_method"] == "LoRA":
        print("Applying LoRA ...")
        lora_cfg = LoraConfig(
            r=cfg["lora_config"]["r"],
            lora_alpha=cfg["lora_config"]["lora_alpha"],
            lora_dropout=cfg["lora_config"]["lora_dropout"],
            target_modules=cfg["lora_config"]["target_modules"],
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    elif cfg["peft_method"] == "DPT":
        print("Applying Deep Prompt Tuning ...")
        v_cfg = model.vision_model.config
        model.config.vocab_size = model.config.text_config.vocab_size
        dpt_cfg = PromptEncoderConfig(
            peft_type="P_TUNING",
            task_type=TaskType.FEATURE_EXTRACTION,
            num_layers=v_cfg.num_hidden_layers,
            num_virtual_tokens=cfg["dpt_config"]["no_learnable_tokens"],
            encoder_reparameterization_type="MLP",
            token_dim=v_cfg.hidden_size,
            num_transformer_submodules=v_cfg.num_hidden_layers,
            num_attention_heads=v_cfg.num_attention_heads,
        )
        model = get_peft_model(model, dpt_cfg)
        model.print_trainable_parameters()
    else:
        print("Full fine-tuning (no PEFT adapter).")

    # ── MLP head ─────────────────────────────────────────────────────────
    mlp = MLP3_Gated(input_dim=args.mlp_input_dim).to(device).to(torch.bfloat16)
    with torch.no_grad():
        for m in mlp.modules():
            if isinstance(m, (ParamSigmoid2, ParamLeakyReLU2)):
                m.float()

    model.requires_grad_(True)
    mlp.requires_grad_(True)

    # ── Dataset / DataLoader ─────────────────────────────────────────────
    train_ds, eval_ds = build_datasets(args.dataset, dataset_paths, Seed)
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        drop_last=True, worker_init_fn=seed_worker,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=cfg["batch_size"], shuffle=False,
        drop_last=True, worker_init_fn=seed_worker,
    )

    # ── Optimizer / Scheduler ────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(mlp.parameters()),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = MultiStepLR(
        optimizer,
        milestones=cfg["lr_scheduler_milestones"],
        gamma=0.2,
    )

    # ── Resume state ─────────────────────────────────────────────────────
    resume_path     = f"resume_state/{cfg['stage_name']}_latest.pt"
    global_step     = 0
    start_epoch     = 0
    best_eval_loss  = float("inf")
    patience_ctr    = 0
    best_SRCC       = float("-inf")
    ckpt            = None

    if cfg["resume"] and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        best_eval_loss = ckpt["best_eval_loss"]
        patience_ctr   = ckpt["patience_counter"]
        start_epoch    = ckpt["epoch"]
        global_step    = ckpt["global_step"]
        best_SRCC      = ckpt.get("best_SRCC", best_SRCC)
        print(f"Resuming from {resume_path} (epoch {start_epoch}, step {global_step})")
    else:
        print("Starting training from scratch.")

    model.train()
    mlp.train()

    # ── Accelerator prepare ──────────────────────────────────────────────
    model, mlp, optimizer, scheduler, train_loader = accelerator.prepare(
        model, mlp, optimizer, scheduler, train_loader,
    )

    if ckpt is not None:
        accelerator.unwrap_model(model).load_state_dict(ckpt["model_state_dict"])
        mlp.load_state_dict(ckpt["mlp_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    # ── WandB ────────────────────────────────────────────────────────────
    use_wandb = not args.no_wandb
    if use_wandb and accelerator.is_main_process:
        wandb.init(
            project=cfg["wandb_project"],
            name=f"{cfg['stage_name']}_{args.dataset}",
        )
        art = wandb.Artifact("source-code", type="code")
        art.add_file(__file__)
        wandb.log_artifact(art)

    train_db = _db_name(train_loader)
    eval_db  = _db_name(eval_loader)
    stage    = cfg["stage_name"]

    # ── Epoch loop ───────────────────────────────────────────────────────
    for epoch in tqdm(range(start_epoch, cfg["epochs"]), desc="Epochs",
                      total=cfg["epochs"], initial=start_epoch):
        dry_cnt = 0
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            total_loss = 0.0
            accum_loss = 0.0
            optimizer.zero_grad()

            for batch in tqdm(train_loader, desc="Training"):
                if cfg["dry_run"]:
                    dry_cnt += 1
                    if dry_cnt >= 100:
                        break

                images = batch["image"].to(device)
                inputs = processor(images=images, return_tensors="pt").to(model.device)

                try:
                    features = model.module.get_image_features(**inputs)
                except Exception:
                    features = model.get_image_features(**inputs)

                score = mlp(features)
                loss_mse    = torch.nn.functional.mse_loss(score.squeeze(1), batch["score"].to(device))
                loss_margin = margin_loss(score.squeeze(1), batch["score"].to(device))
                loss = (loss_mse + loss_margin) / cfg["gradient_accumulation_steps"]

                accelerator.backward(loss)
                accum_loss += loss.item()
                total_loss += loss.item() * cfg["gradient_accumulation_steps"]
                global_step += 1

                # Gate weight logging
                if use_wandb and accelerator.is_main_process:
                    try:
                        w_mean = torch.sigmoid(mlp.module.act1.g).mean().item()
                    except AttributeError:
                        w_mean = torch.sigmoid(mlp.act1.g).mean().item()
                    wandb.log({"gate/w_mean": w_mean}, commit=False)

                if global_step % cfg["gradient_accumulation_steps"] == 0:
                    if cfg["use_gradient_clip"]:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
                    optimizer.step()
                    optimizer.zero_grad()
                    tqdm.write(f"Step {global_step} — accum loss: {accum_loss:.4f}")
                    if use_wandb and accelerator.is_main_process:
                        wandb.log({f"{stage}_accumulated_loss": accum_loss}, commit=False)
                    accum_loss = 0.0

                # Periodic checkpoint
                if global_step % cfg["checkpoint_steps"] == 0 and accelerator.is_main_process:
                    os.makedirs("checkpoints", exist_ok=True)
                    ckpt_dir = f"checkpoints/{stage}_step_train_{train_db}_Test{eval_db}_{global_step}/"
                    accelerator.unwrap_model(model).save_pretrained(ckpt_dir)
                    torch.save(mlp.state_dict(), f"{ckpt_dir}/mlp.pt")
                    _clean_old_checkpoints(stage, cfg["max_checkpoints"])

                # Per-step logging
                if use_wandb and accelerator.is_main_process:
                    wandb.log({
                        f"{stage}_loss":  loss.item(),
                        f"{stage}_epoch": epoch + 1,
                        f"{stage}_step":  global_step,
                        f"{stage}_lr":    optimizer.param_groups[0]["lr"],
                    })

            # End of epoch
            avg = total_loss / max(len(train_loader), 1)
            if use_wandb and accelerator.is_main_process:
                wandb.log({f"{stage}_total_loss": avg, f"{stage}_epoch": epoch + 1}, commit=False)
            print(f"Epoch {epoch+1}/{cfg['epochs']} — avg loss: {avg:.4f}")

            if cfg["lr_scheduler"]:
                scheduler.step()

            # ── Evaluation ───────────────────────────────────────────────
            SRCC = float("-inf")
            if cfg["do_eval"] and epoch % cfg["eval_epoch_steps"] == 0:
                results, avg_eval = evaluate(
                    model, mlp, processor, eval_loader, device,
                    dry_run=cfg["dry_run"],
                )
                SRCC = results["SRCC"]
                PLCC = results["PLCC"]
                if accelerator.is_main_process:
                    tqdm.write(f"  SRCC: {SRCC:.4f}  PLCC: {PLCC:.4f}  eval_loss: {avg_eval:.4f}")
                    if use_wandb:
                        wandb.log({
                            f"{stage}_SRCC": SRCC,
                            f"{stage}_PLCC": PLCC,
                            f"{stage}_Avg_Eval_Loss": avg_eval,
                        })

            # ── Best checkpoint ───────────────────────────────────────────
            if accelerator.is_main_process and SRCC > best_SRCC:
                best_SRCC = SRCC
                os.makedirs("best_checkpoints", exist_ok=True)
                best_dir = f"best_checkpoints/{stage}_train_{train_db}_test_{eval_db}"
                accelerator.unwrap_model(model).save_pretrained(best_dir)
                torch.save(mlp.state_dict(), f"{best_dir}/mlp.pt")
                tqdm.write(f"  [Best] SRCC={best_SRCC:.4f} saved to {best_dir}")

            # ── Resume state at epoch end ─────────────────────────────────
            if accelerator.is_main_process:
                os.makedirs("resume_state", exist_ok=True)
                torch.save({
                    "epoch":              epoch + 1,
                    "global_step":        global_step,
                    "model_state_dict":   accelerator.unwrap_model(model).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_eval_loss":     best_eval_loss,
                    "patience_counter":   patience_ctr,
                    "mlp_state_dict":     mlp.state_dict(),
                    "best_SRCC":          best_SRCC,
                }, resume_path)

    # ── Final evaluation ─────────────────────────────────────────────────
    if cfg["do_eval"]:
        print("Final evaluation ...")
        results, avg_eval = evaluate(model, mlp, processor, eval_loader, device, dry_run=cfg["dry_run"])
        print(f"  SRCC: {results['SRCC']:.4f}  PLCC: {results['PLCC']:.4f}  loss: {avg_eval:.4f}")
        if use_wandb and accelerator.is_main_process:
            wandb.log({f"{stage}_SRCC": results["SRCC"], f"{stage}_PLCC": results["PLCC"]})

    # ── Save final checkpoint ────────────────────────────────────────────
    if accelerator.is_main_process:
        os.makedirs("checkpoints", exist_ok=True)
        final_dir = f"checkpoints/{stage}_final_train_{train_db}_Test{eval_db}/"
        accelerator.unwrap_model(model).save_pretrained(final_dir)
        torch.save(mlp.state_dict(), f"{final_dir}/mlp.pt")

        os.makedirs("results", exist_ok=True)
        res_path = f"results/results_{stage}_Train_{train_db}_Test_{eval_db}.json"
        if cfg["do_eval"]:
            with open(res_path, "w") as f:
                json.dump(results, f, indent=4)
            print(f"Results saved to {res_path}")

    if use_wandb and accelerator.is_main_process:
        wandb.finish()

    accelerator.free_memory()
    torch.cuda.empty_cache()
    print("Training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train NR-IQA AGM model")

    # Dataset
    p.add_argument("--dataset", type=str, required=True,
                   help="Dataset id: CLIVE, KonIQ_10K, SPAQ, KADID10K, FLIVE, "
                        "AGIQA3K, AGIQA1K, KonIQ_10K_CLIVE, CLIVE_KonIQ_10K")
    p.add_argument("--data_dir", type=str, default="./Dataset",
                   help="Root directory containing all dataset folders")

    # Model
    p.add_argument("--model_id", type=str, default=MODEL_CONFIG["model_id"])
    p.add_argument("--mlp_input_dim", type=int, default=MODEL_CONFIG["mlp_input_dim"])

    # PEFT
    p.add_argument("--peft_method", type=str, default=TRAIN_CONFIG["peft_method"],
                   choices=["LoRA", "DPT", "NA"])
    p.add_argument("--lora_r", type=int, default=None)
    p.add_argument("--lora_alpha", type=int, default=8)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # Training
    p.add_argument("--epochs", type=int, default=TRAIN_CONFIG["epochs"])
    p.add_argument("--batch_size", type=int, default=TRAIN_CONFIG["batch_size"])
    p.add_argument("--lr", type=float, default=TRAIN_CONFIG["learning_rate"])
    p.add_argument("--weight_decay", type=float, default=TRAIN_CONFIG["weight_decay"])
    p.add_argument("--grad_accum", type=int, default=TRAIN_CONFIG["gradient_accumulation_steps"])
    p.add_argument("--gradient_clip", type=float, default=0.0,
                   help="Max gradient norm; 0 disables clipping")

    # Scheduler
    p.add_argument("--no_scheduler", action="store_true")
    p.add_argument("--lr_milestones", type=str, default="30,35",
                   help="Comma-separated epoch milestones for MultiStepLR")

    # Checkpointing / resuming
    p.add_argument("--checkpoint_steps", type=int, default=TRAIN_CONFIG["checkpoint_steps"])
    p.add_argument("--max_checkpoints", type=int, default=TRAIN_CONFIG["max_checkpoints"])
    p.add_argument("--stage_name", type=str, default=f"AGM_seed{Seed}")
    p.add_argument("--resume", action="store_true")

    # Eval
    p.add_argument("--no_eval", action="store_true")
    p.add_argument("--eval_every", type=int, default=TRAIN_CONFIG["eval_epoch_steps"],
                   help="Run evaluation every N epochs")

    # Logging
    p.add_argument("--wandb_project", type=str, default=TRAIN_CONFIG["wandb_project"])
    p.add_argument("--no_wandb", action="store_true")

    # Debug
    p.add_argument("--dry_run", action="store_true",
                   help="Only run 100 train batches and 32 eval batches per epoch")

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
