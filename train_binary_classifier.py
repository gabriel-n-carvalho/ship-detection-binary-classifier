# train_binary_classifier.py
# Deterministic training + CPU-only Docker config + Early Stopping
# - Full run-to-run reproducibility (same seeds => same metrics & checkpoints)
# - Early stopping on validation PR-AUC with patience, robust to NaN PR-AUC
# - Keeps best-on-PR-AUC and always saves last checkpoint
# - Reproducible resume (saves/loads RNG state)
# - Optimized for CPU-only Docker execution
# ---------------------------------------------------------------------------

# ===== Determinism & threading (MUST run before importing torch/numpy) =====
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score
from torchvision.transforms import InterpolationMode
from torchvision import transforms
import timm  # PyTorch Image Models library

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
from PIL import Image, ImageFile
import pandas as pd
import numpy as np
from datetime import timedelta
import time
import sys
import math
import random
import warnings
from pathlib import Path
import argparse
import os

# Ensure deterministic hashing/ordering in Python
# This prevents Python's hash randomization from affecting reproducibility
os.environ.setdefault("PYTHONHASHSEED", "0")

# CPU-specific optimizations for Docker
# Calculate optimal number of threads, leaving one core free for system processes
_omp = str(max(1, (os.cpu_count() or 2) - 1))
os.environ.setdefault("OMP_NUM_THREADS", _omp)  # OpenMP threading
os.environ.setdefault("MKL_NUM_THREADS", _omp)  # Intel MKL threading
os.environ.setdefault("NUMEXPR_NUM_THREADS", _omp)  # NumExpr threading
os.environ.setdefault("BLAS_NUM_THREADS", _omp)  # BLAS threading

# Disable CUDA/MPS completely for CPU-only execution
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # Disable CUDA
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # Disable MPS

# =============================== Imports ===================================
# Standard library imports

# Data science and numerical computing libraries

# Deep learning framework

# Computer vision and model libraries

# Progress logging with tqdm

# Silence a noisy urllib3 warning, if available
# This prevents SSL-related warnings that can clutter the output
try:
    from urllib3.exceptions import NotOpenSSLWarning
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except Exception:
    pass

# =========================== Paths & hyperparams ============================
# Data paths - assumes specific directory structure
# Directory containing training images
IMG_DIR = Path("airbus-ship-detection/train_v2")
TRAIN_CSV = "labels/splits/train.csv"  # Training data CSV with image IDs and labels
VAL_CSV = "labels/splits/val.csv"      # Validation data CSV with image IDs and labels

# Model and training hyperparameters (defaults, can be overridden by CLI)
# Input image size (will be resized to this)
IMG_SIZE = 256
BATCH_SIZE = 32
# Maximum number of training epochs
EPOCHS = 100
LR = 3e-4                 # Learning rate
# Random seed for reproducibility
SEED = 42

# Early stopping configuration
# stop if no PR-AUC improvement for this many epochs after first valid PR-AUC
EARLY_STOP_PATIENCE = 20

# Model architecture
MODEL_NAME = "tf_efficientnetv2_b0.in1k"  # EfficientNet-B0 model from timm

# Output directories
# Directory for saving model checkpoints
OUTDIR = Path("outputs/models")
OUTDIR.mkdir(exist_ok=True, parents=True)

# Gradient accumulation (for effective larger batch sizes)
GRAD_ACCUM_STEPS = 1

# Subsetting configuration (for faster development/testing)
SUBSET_FRACTION = None  # Fraction of data to use (None = use all)
SUBSET_SIZE = None     # Number of samples to use (disabled by default)

# =============================== Misc setup ================================
# Allow loading truncated images (useful for corrupted files)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Suppress PyTorch warnings for cleaner output
torch.set_warn_always(False)

# Force CPU-only execution
DEVICE = "cpu"
print(f"CPU-only execution enforced. Using device: {DEVICE}")


def set_seed(s: int):
    """
    Set random seeds for all libraries to ensure reproducibility.

    Args:
        s: Seed value
    """
    # Set seeds for all random number generators
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)

    # CPU matmul precision (doesn't break determinism)
    # Use high precision for CPU matrix multiplications
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # Hard fail on any non-deterministic kernels
    # This ensures complete reproducibility by failing if non-deterministic ops are used
    torch.use_deterministic_algorithms(True, warn_only=False)


# Set PyTorch threads (after import; env for OMP/MKL was set before import)
# Configure PyTorch to use optimal number of CPU threads
try:
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    print(f"Set PyTorch to use {torch.get_num_threads()} threads")
except Exception:
    pass

# ================================ Dataset ==================================


class AirbusBinaryDS(Dataset):
    """
    Custom PyTorch Dataset for Airbus ship detection binary classification.

    This dataset loads images and their corresponding binary labels (ship/no ship)
    and applies appropriate transformations for training or validation.
    """

    def __init__(self, df: pd.DataFrame, img_dir: Path, train: bool = True):
        """
        Initialize the dataset.

        Args:
            df: DataFrame containing image IDs and labels
            img_dir: Directory containing the images
            train: Whether this is for training (affects data augmentation)
        """
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)

        # RGB fill color for image transformations (ImageNet mean)
        fill_rgb = tuple(int(c * 255) for c in (0.485, 0.456, 0.406))

        if train:
            # Training transforms with data augmentation
            self.tf = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE),
                                  interpolation=InterpolationMode.BILINEAR),
                transforms.RandomHorizontalFlip(),      # Random horizontal flip
                # Random vertical flip (20% chance)
                transforms.RandomVerticalFlip(p=0.2),
                # Random rotation ±5 degrees
                transforms.RandomRotation(5, fill=fill_rgb),
                transforms.ColorJitter(
                    0.05, 0.05, 0.05, 0.02),  # Color jittering
                transforms.ToTensor(),                  # Convert to tensor
                transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet normalization
                                     [0.229, 0.224, 0.225]),
            ])
        else:
            # Validation transforms (no augmentation)
            self.tf = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE),
                                  interpolation=InterpolationMode.BILINEAR),
                transforms.ToTensor(),                  # Convert to tensor
                transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet normalization
                                     [0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, i):
        """
        Get a single sample from the dataset.

        Args:
            i: Index of the sample

        Returns:
            tuple: (image_tensor, label_tensor)
        """
        r = self.df.iloc[i]
        img_path = self.img_dir / r.ImageId

        # Load and convert image to RGB
        try:
            with Image.open(img_path) as im:
                img = im.convert("RGB")
        except Exception:
            # Create black image if loading fails
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))

        # Apply transformations
        x = self.tf(img)

        # Convert label to tensor
        y = torch.tensor([float(r.has_ship)], dtype=torch.float32)
        return x, y


# ===================== Deterministic DataLoader helpers =====================

def _seed_worker(worker_id: int):
    """
    Set random seed for DataLoader worker processes.

    This ensures each worker has a different but deterministic seed,
    preventing race conditions while maintaining reproducibility.

    Each worker gets a unique seed derived from the global SEED + worker_id,
    ensuring deterministic but different random sequences across workers.

    Args:
        worker_id: ID of the worker process
    """
    worker_seed = (SEED + worker_id) % (2**32 - 1)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


_DATA_GENERATOR = None


def _get_generator():
    """
    Get or create a deterministic PyTorch generator for DataLoader shuffling.

    Returns:
        torch.Generator: Deterministic random generator
    """
    global _DATA_GENERATOR
    if _DATA_GENERATOR is None:
        g = torch.Generator()
        g.manual_seed(SEED)
        _DATA_GENERATOR = g
    return _DATA_GENERATOR


# ============================== Training utils ==============================

def move_to_device_batch(xb, yb, device: str):
    """
    Move batch data to the specified device.

    Args:
        xb: Input batch
        yb: Target batch
        device: Target device

    Returns:
        tuple: (xb_on_device, yb_on_device)
    """
    xb = xb.to(device, non_blocking=False)  # CPU doesn't need non_blocking
    yb = yb.to(device, non_blocking=False)
    return xb, yb


def create_loaders(train_df, val_df, device: str):
    """
    Create training and validation data loaders.

    Args:
        train_df: Training data DataFrame
        val_df: Validation data DataFrame
        device: Device string (always 'cpu' for this script)

    Returns:
        tuple: (train_dataset, val_dataset, train_loader, val_loader)
    """
    # Create datasets
    tr_ds = AirbusBinaryDS(train_df, IMG_DIR, train=True)
    va_ds = AirbusBinaryDS(val_df, IMG_DIR, train=False)

    # DataLoader configuration with shared memory safety
    # Try to detect if we're in a Docker container with limited shared memory
    try:
        # Check if we're in Docker by looking for common Docker environment variables
        in_docker = os.path.exists('/.dockerenv') or 'DOCKER' in os.environ

        if in_docker:
            # Very conservative settings for Docker containers
            num_workers = 1  # Single worker to avoid shared memory issues
            persistent_workers = False  # Disable persistent workers in Docker
            print(
                "Docker environment detected - using single worker to avoid shared memory issues")
        else:
            # Normal settings for non-Docker environments
            num_workers = max(1, min(2, (os.cpu_count() or 2) - 1))
            persistent_workers = True
            print(f"Non-Docker environment - using {num_workers} workers")

    except Exception as e:
        # Fallback to safest settings
        print(f"Error detecting environment, using fallback settings: {e}")
        num_workers = 1
        persistent_workers = False

    dl_kwargs = dict(
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        pin_memory=False,  # No pin memory for CPU
        persistent_workers=persistent_workers,
        drop_last=False,
        worker_init_fn=_seed_worker,  # Set worker seeds (ensures determinism)
        generator=_get_generator(),   # Deterministic shuffling
    )

    # Create loaders
    tr_loader = DataLoader(tr_ds, shuffle=True, **dl_kwargs)
    va_loader = DataLoader(va_ds, shuffle=False, **dl_kwargs)
    return tr_ds, va_ds, tr_loader, va_loader


def create_model(device: str):
    """
    Create and initialize the model.

    Args:
        device: Device to place the model on (always 'cpu' for this script)

    Returns:
        nn.Module: Initialized model
    """
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=1)
    model = model.to(device)
    return model


# ===== Reproducible resume: save/restore full RNG state (python, numpy, torch, dataloader) =====

def _get_rng_state():
    """
    Capture the complete random number generator state for all libraries.

    This is used for reproducible checkpoint resuming.

    Returns:
        dict: Complete RNG state for all libraries
    """
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "dl_gen": _get_generator().get_state(),
    }
    return state


def _set_rng_state(state):
    """
    Restore the complete random number generator state for all libraries.

    Args:
        state: Complete RNG state dictionary
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    _get_generator().set_state(state["dl_gen"])


def train_one_fold(train_df, val_df, fold=0, resume_from_checkpoint=None):
    """
    Train a model for one fold with early stopping and checkpointing.

    Args:
        train_df: Training data DataFrame
        val_df: Validation data DataFrame
        fold: Fold number (for logging and checkpoint naming)
        resume_from_checkpoint: Path to checkpoint to resume from

    Returns:
        Path: Path to the best model checkpoint
    """
    # Use CPU device
    device = DEVICE
    print(f"Using device: {device}")

    # Clean up memory
    import gc
    gc.collect()

    # --- sanity checks for degenerate labels ---
    # Ensure we have both positive and negative samples
    pos = float(train_df.has_ship.sum())
    neg = float(len(train_df) - pos)
    if pos == 0 or neg == 0:
        raise ValueError(
            f"Training labels are degenerate: pos={pos:.0f}, neg={neg:.0f}. "
            "Need both classes to train and compute AUC."
        )

    # Create data loaders and model
    tr_ds, va_ds, tr_loader, va_loader = create_loaders(
        train_df, val_df, device)
    model = create_model(device)

    # Class imbalance handling
    # Calculate positive weight to handle imbalanced classes
    pos_weight = torch.tensor([max(1.0, neg / max(1.0, pos))],
                              device=device, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS)

    # Early stopping variables
    best_pr_auc = float("nan")
    best_epoch = 0
    no_improve = 0

    # Setup checkpoint and logging paths
    best_path = OUTDIR / f"best_fold{fold}.pt"

    # Resume from checkpoint if provided
    start_epoch = 1
    if resume_from_checkpoint is not None:
        checkpoint_path = Path(resume_from_checkpoint)
        if checkpoint_path.exists():
            print(f"Loading checkpoint from {checkpoint_path}")

            # Try loading with weights_only=False first (for older checkpoints)
            try:
                checkpoint = torch.load(
                    checkpoint_path, map_location=device, weights_only=False)
            except Exception as e:
                print(
                    f"Failed to load checkpoint with weights_only=False: {e}")
                print("Trying with weights_only=True...")
                try:
                    checkpoint = torch.load(
                        checkpoint_path, map_location=device, weights_only=True)
                except Exception as e2:
                    print(f"Failed to load checkpoint: {e2}")
                    print("Starting from scratch.")
                    checkpoint = None

            if checkpoint is not None:
                # Load model state
                model.load_state_dict(checkpoint['model'])

                # Load optimizer/scheduler state if available
                if 'optimizer' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                if 'scheduler' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler'])

                # Load training state
                if 'epoch' in checkpoint:
                    start_epoch = int(checkpoint['epoch']) + 1
                    print(f"Resuming from epoch {start_epoch}")

                # Handle both old (auc) and new (pr_auc) checkpoint formats
                if 'pr_auc' in checkpoint:
                    best_pr_auc = float(checkpoint['pr_auc'])
                    print(f"Previous best PR-AUC: {best_pr_auc:.4f}" if not math.isnan(
                        best_pr_auc) else "Previous best PR-AUC: NaN")
                elif 'auc' in checkpoint:
                    # Convert old ROC-AUC checkpoint to PR-AUC format
                    old_auc = float(checkpoint['auc'])
                    # Reset PR-AUC since we don't have it
                    best_pr_auc = float("nan")
                    print(f"Found old checkpoint with ROC-AUC: {old_auc:.4f}")
                    print("Converting to PR-AUC format - PR-AUC will be recalculated.")

                # Restore RNG state for reproducible continuation
                if 'rng_state' in checkpoint:
                    try:
                        _set_rng_state(checkpoint['rng_state'])
                        print("RNG state restored for reproducible resume.")
                    except Exception as e:
                        print(f"Warning: Could not restore RNG state: {e}")

                print("Checkpoint loaded successfully!")
            else:
                print("Failed to load checkpoint. Starting from scratch.")
        else:
            print(
                f"Warning: Checkpoint {checkpoint_path} not found. Starting from scratch.")

    # ----- Training loop -----
    global_step = 0
    val_global_step = 0
    prev_avg_train_batch_s = None
    prev_avg_val_batch_s = None
    for epoch in range(start_epoch, EPOCHS + 1):
        print(f"\n{'='*60}")
        print(f"Starting Epoch {epoch:02d}/{EPOCHS}")
        print(
            f"Training batches: {len(tr_loader)}, Validation batches: {len(va_loader)}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"Gradient accumulation steps: {GRAD_ACCUM_STEPS}")
        print(f"Effective batch size: {BATCH_SIZE * GRAD_ACCUM_STEPS}")
        print(f"{'='*60}")

        # Training phase
        model.train()
        tr_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        # Iterate through training batches with tqdm progress bar
        total_batches = len(tr_loader)
        if prev_avg_train_batch_s is not None and total_batches > 0:
            est_train_secs = prev_avg_train_batch_s * total_batches
            print(
                f"Estimated train epoch duration: ~{str(timedelta(seconds=int(est_train_secs)))}", flush=True)
        epoch_train_start = time.time()
        loop = tqdm(
            tr_loader,
            total=total_batches,
            desc=f"Epoch {epoch:02d}/{EPOCHS}",
            unit="batch",
            file=sys.stdout,
            dynamic_ncols=True,
            leave=False,
            ascii=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {percentage:3.0f}% ETA {remaining}",
        )
        train_est_printed_from_live = False
        for step, (xb, yb) in enumerate(loop, start=1):
            # Move data to device
            xb, yb = move_to_device_batch(xb, yb, device)

            # Forward pass
            logits = model(xb).squeeze(-1)
            loss = criterion(logits, yb.squeeze(-1)) / GRAD_ACCUM_STEPS

            # Backward pass
            loss.backward()

            # Calculate current loss for progress bar
            current_loss = loss.item() * GRAD_ACCUM_STEPS

            # Gradient accumulation and optimization
            if (step % GRAD_ACCUM_STEPS) == 0:
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Optimizer step
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # Accumulate loss for logging
            tr_loss += (loss.item() * xb.size(0) * GRAD_ACCUM_STEPS)
            global_step += 1

            # Update progress bar postfix each iteration
            loop.set_postfix(loss=f"{current_loss:.4f}")
            # For first epoch, print estimated duration after a few batches
            if not train_est_printed_from_live and step >= max(5, total_batches // 100):
                elapsed = time.time() - epoch_train_start
                avg_per_batch = elapsed / max(1, step)
                est_total = avg_per_batch * total_batches
                print(
                    f"Estimated train epoch duration: ~{str(timedelta(seconds=int(est_total)))}", flush=True)
                train_est_printed_from_live = True
        loop.close()
        # Save average train batch time for next epoch's early estimate
        train_elapsed_total = max(0.0, time.time() - epoch_train_start)
        if total_batches > 0:
            prev_avg_train_batch_s = train_elapsed_total / total_batches

        # Calculate average training loss
        tr_loss /= len(tr_ds)
        print(f"Training completed - Average loss: {tr_loss:.4f}")

        # ---- Validation phase ----
        print(f"\nStarting validation...")
        model.eval()
        val_loss = 0.0
        probs_list, y_list = [], []

        # Disable gradient computation for validation
        with torch.inference_mode():
            total_val_batches = len(va_loader)
            if prev_avg_val_batch_s is not None and total_val_batches > 0:
                est_val_secs = prev_avg_val_batch_s * total_val_batches
                print(
                    f"Estimated val epoch duration:   ~{str(timedelta(seconds=int(est_val_secs)))}", flush=True)
            epoch_val_start = time.time()
            loop_val = tqdm(
                va_loader,
                total=total_val_batches,
                desc=f"Val   {epoch:02d}/{EPOCHS}",
                unit="batch",
                file=sys.stdout,
                dynamic_ncols=True,
                leave=False,
                ascii=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {percentage:3.0f}% ETA {remaining}",
            )
            val_est_printed_from_live = False
            for step, (xb, yb) in enumerate(loop_val, start=1):
                xb, yb = move_to_device_batch(xb, yb, device)

                # Forward pass
                logits = model(xb).squeeze(-1)
                loss = criterion(logits, yb.squeeze(-1))

                # Accumulate validation loss and predictions
                val_loss += loss.item() * xb.size(0)
                probs_list.append(torch.sigmoid(logits).detach().to(
                    dtype=torch.float32, device="cpu").numpy())
                y_list.append(yb.squeeze(-1).cpu().numpy())
                val_global_step += 1

                # Update progress bar postfix each iteration
                loop_val.set_postfix(val_loss=f"{loss.item():.4f}")
                # For first epoch, print estimated duration after a few batches
                if not val_est_printed_from_live and step >= max(5, total_val_batches // 100):
                    elapsed_val = time.time() - epoch_val_start
                    avg_per_batch_val = elapsed_val / max(1, step)
                    est_total_val = avg_per_batch_val * total_val_batches
                    print(
                        f"Estimated val epoch duration:   ~{str(timedelta(seconds=int(est_total_val)))}", flush=True)
                    val_est_printed_from_live = True
            loop_val.close()
            # Save average val batch time for next epoch's early estimate
            val_elapsed_total = max(0.0, time.time() - epoch_val_start)
            if total_val_batches > 0:
                prev_avg_val_batch_s = val_elapsed_total / total_val_batches

        # Calculate validation metrics
        val_loss /= len(va_ds)
        print(f"Validation completed - Average loss: {val_loss:.4f}")
        probs = np.concatenate(probs_list) if len(probs_list) else np.array([])
        y_true = np.concatenate(y_list) if len(y_list) else np.array([])

        # Compute classification metrics
        if probs.size > 0:
            y_pred = (probs >= 0.5).astype(np.uint8)  # Binary predictions
            acc = accuracy_score(y_true, y_pred)
            p, r, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0)
            try:
                roc_auc = roc_auc_score(y_true, probs)  # ROC-AUC score
            except ValueError:
                # Handle edge cases (e.g., only one class)
                roc_auc = float("nan")
            try:
                pr_auc = average_precision_score(y_true, probs)  # PR-AUC score
            except ValueError:
                # Handle edge cases (e.g., only one class)
                pr_auc = float("nan")
        else:
            acc = p = r = f1 = float("nan")
            roc_auc = float("nan")
            pr_auc = float("nan")

        # Print epoch results
        print(f"[Fold {fold}] Epoch {epoch:02d}/{EPOCHS} | "
              f"train_loss {tr_loss:.4f} | val_loss {val_loss:.4f} | "
              f"ACC {acc:.4f} P {p:.4f} R {r:.4f} F1 {f1:.4f} ROC-AUC {roc_auc:.4f} PR-AUC {pr_auc:.4f}")

        # Get current learning rate for logging
        current_lr = optimizer.param_groups[0]["lr"]

        # Step the learning rate scheduler
        scheduler.step()
        print(
            f"Learning rate updated to: {optimizer.param_groups[0]['lr']:.6f}")

        # ---- Checkpointing (best & last) + Early Stopping tracking ----
        improved = False
        if not math.isnan(pr_auc):
            # Save best model based on PR-AUC
            if math.isnan(best_pr_auc) or (pr_auc > best_pr_auc):
                best_pr_auc = pr_auc
                best_epoch = epoch
                no_improve = 0
                improved = True
                torch.save({"model": model.state_dict(),
                            "pr_auc": float(best_pr_auc),
                            "model_name": MODEL_NAME,
                            "img_size": IMG_SIZE},
                           best_path, _use_new_zipfile_serialization=False)

        # Always save last checkpoint to allow resuming (with RNG state)
        checkpoint_data = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "pr_auc": float(best_pr_auc) if not math.isnan(best_pr_auc) else float("nan"),
            "model_name": MODEL_NAME,
            "img_size": IMG_SIZE,
            "rng_state": _get_rng_state(),  # Save RNG state for reproducible resume
        }

        torch.save(checkpoint_data, OUTDIR /
                   f"last_fold{fold}.pt", _use_new_zipfile_serialization=False)

        # Update learning rate
        scheduler.step()

        # Early stopping: only start patience after first valid PR-AUC has been seen
        if not math.isnan(best_pr_auc):
            if not improved:
                no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch} "
                      f"(no PR-AUC improvement for {EARLY_STOP_PATIENCE} epochs).")
                break

    # Guarantee a 'best' checkpoint exists even if PR-AUC never became valid
    if not best_path.exists():
        torch.save({"model": model.state_dict(),
                    "pr_auc": float(best_pr_auc) if not math.isnan(best_pr_auc) else float("nan"),
                    "model_name": MODEL_NAME, "img_size": IMG_SIZE},
                   best_path, _use_new_zipfile_serialization=False)

    # Print final results
    print(f"Best PR-AUC fold {fold}: {best_pr_auc:.4f}" if not math.isnan(best_pr_auc)
          else f"Best PR-AUC fold {fold}: NaN (no valid PR-AUC computed)")
    print(f"Best checkpoint saved to: {best_path}")
    return best_path


# ================================== Main ===================================

def main():
    """
    Main function that orchestrates the training process.

    This function:
    1. Parses command line arguments
    2. Sets up reproducibility
    3. Loads and preprocesses data
    4. Optionally subsets data for faster development
    5. Trains the model with checkpoint resuming
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train binary classifier with deterministic settings')
    parser.add_argument('--fold', type=int, default=0,
                        help='Fold number (used in checkpoint filenames)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--img-size', type=int, default=256,
                        help='Input image size (will be resized to this)')
    parser.add_argument('--grad-accum-steps', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--early-stop-patience', type=int, default=20,
                        help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--model-name', type=str, default='tf_efficientnetv2_b0.in1k',
                        help='Model architecture name from timm')
    parser.add_argument('--subset-fraction', type=str, default=None, required=False,
                        help='Fraction of data to use (0.0-1.0, empty = use all)')
    parser.add_argument('--subset-size', type=str, default=None, required=False,
                        help='Number of samples to use (empty = use all)')
    args = parser.parse_args()

    # Update global variables with CLI arguments
    global SEED, BATCH_SIZE, EPOCHS, LR, IMG_SIZE, GRAD_ACCUM_STEPS, EARLY_STOP_PATIENCE, MODEL_NAME, SUBSET_FRACTION, SUBSET_SIZE
    SEED = args.seed
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LR = args.lr
    IMG_SIZE = args.img_size
    GRAD_ACCUM_STEPS = args.grad_accum_steps
    EARLY_STOP_PATIENCE = args.early_stop_patience
    MODEL_NAME = args.model_name
    
    # Handle subset parameters (convert to appropriate types if provided)
    SUBSET_FRACTION = float(args.subset_fraction) if args.subset_fraction and args.subset_fraction.strip() else None
    SUBSET_SIZE = int(args.subset_size) if args.subset_size and args.subset_size.strip() else None
    
    set_seed(SEED)

    # Load pre-split train and validation data
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)

    # Ensure label dtype is stable (prevent type issues)
    train_df["has_ship"] = train_df["has_ship"].astype(np.float32)
    val_df["has_ship"] = val_df["has_ship"].astype(np.float32)

    # Print dataset information
    print(
        f"Loaded train: {len(train_df)} rows, class_counts={train_df['has_ship'].value_counts().to_dict()}")
    print(
        f"Loaded val:   {len(val_df)} rows, class_counts={val_df['has_ship'].value_counts().to_dict()}")

    # Optional deterministic subsetting (stratified) for quicker iterations
    # This allows using a subset of data for faster development/testing
    if (SUBSET_FRACTION is not None) or (SUBSET_SIZE is not None):
        n_total_train = len(train_df)
        n_total_val = len(val_df)

        # Calculate subset sizes
        if SUBSET_FRACTION is not None:
            k_train = max(
                1, int(round(n_total_train * float(SUBSET_FRACTION))))
            k_val = max(1, int(round(n_total_val * float(SUBSET_FRACTION))))
        else:
            k_train = max(1, int(SUBSET_SIZE))
            val_ratio = n_total_val / max(1, n_total_train)
            k_val = max(1, int(round(k_train * val_ratio)))

        # Stratified sampling (deterministic via SEED)
        # This ensures we maintain class balance in the subset
        def stratified_take(df, k_target):
            """
            Take a stratified subset of the dataframe.

            Args:
                df: DataFrame to subset
                k_target: Target number of samples

            Returns:
                DataFrame: Stratified subset
            """
            frac = k_target / len(df)
            cls_counts = df.groupby('has_ship').size().to_dict()
            per_class_k = {c: max(1, int(round(frac * cnt)))
                           for c, cnt in cls_counts.items()}
            parts = []
            rng = np.random.RandomState(SEED)
            for c, cnt in cls_counts.items():
                take = min(cnt, per_class_k[c])
                idx = df.index[df.has_ship == c].values
                pick = rng.choice(idx, size=take, replace=False)
                parts.append(df.loc[pick])
            out = pd.concat(parts, axis=0).sample(
                frac=1.0, random_state=SEED).reset_index(drop=True)
            return out

        # Apply subsetting
        train_df = stratified_take(train_df, k_train)
        val_df = stratified_take(val_df,   k_val)

        # Print subsetting information
        print("Subsetting enabled:")
        print(
            f"  Train: kept {len(train_df)} rows (target={k_train}). Class counts: {train_df.groupby('has_ship').size().to_dict()}")
        print(
            f"  Val:   kept {len(val_df)} rows (target={k_val}). Class counts: {val_df.groupby('has_ship').size().to_dict()}")

    # Resume from checkpoint if available
    resume_checkpoint = f"{OUTDIR}/last_fold{args.fold}.pt"
    _ = train_one_fold(train_df, val_df, fold=args.fold,
                       resume_from_checkpoint=resume_checkpoint)


if __name__ == "__main__":
    main()
