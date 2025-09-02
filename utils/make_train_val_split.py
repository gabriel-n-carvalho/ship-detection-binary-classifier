#!/usr/bin/env python3
"""
Deterministic, stratified (per-class) hash split for Airbus-style labels.

Input CSV must have:
  - ImageId (string-like)
  - has_ship (0/1)

Optional:
  - You may pass --salt to change the split deterministically (like a seed).

Outputs:
  - train.csv and val.csv (filenames configurable)
"""

import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Constants
DEFAULT_Z_SCORE = 2.58  # 99% confidence interval
HASH_MODULUS = 2**64
DEFAULT_VAL_SIZE = 0.2
DEFAULT_OUTDIR = "labels/splits"
DEFAULT_TRAIN_NAME = "train.csv"
DEFAULT_VAL_NAME = "val.csv"
DEFAULT_LABELS_CSV = "labels/airbus_binary_labels.csv"


def parse_args():
    p = argparse.ArgumentParser(
        description="Deterministic per-class hash split (stratified)."
    )
    p.add_argument(
        "labels_csv",
        nargs="?",
        default=DEFAULT_LABELS_CSV,
        help="CSV with columns: ImageId,has_ship (0/1).",
    )
    p.add_argument(
        "--val-size",
        type=float,
        default=DEFAULT_VAL_SIZE,
        help="Validation fraction in (0,1). Default: 0.2",
    )
    p.add_argument(
        "--salt",
        type=str,
        default="",
        help="Optional string salt to deterministically alter the split.",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default=DEFAULT_OUTDIR,
        help="Output directory for CSVs.",
    )
    p.add_argument(
        "--train-name",
        type=str,
        default=DEFAULT_TRAIN_NAME,
        help="Output filename for train split.",
    )
    p.add_argument(
        "--val-name",
        type=str,
        default=DEFAULT_VAL_NAME,
        help="Output filename for val split.",
    )
    p.add_argument(
        "--sort-output",
        action="store_true",
        help="Sort splits by ImageId before saving (diff-friendly).",
    )
    p.add_argument(
        "--fail-if-exists",
        action="store_true",
        help="Fail if output files already exist (no overwrite).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging.",
    )
    return p.parse_args()


def compute_hash_values(image_ids: pd.Series, salt: str) -> pd.Series:
    """
    Compute deterministic, uniform hash values in [0,1) per ImageId
    using pandas.util.hash_pandas_object (fast, stable).
    """
    salted = (image_ids.astype("string") + "|" + salt).astype("category")
    h = pd.util.hash_pandas_object(salted, index=False).astype("uint64")
    return h / np.float64(HASH_MODULUS)


def stat_within_tolerance(n: int, p_hat: float, p: float, z: float = DEFAULT_Z_SCORE) -> bool:
    """
    Check if observed proportion p_hat is within a z-score confidence interval
    around true proportion p. Default z≈2.58 (~99%).
    """
    if n <= 0:
        return False
    se = math.sqrt(max(p * (1 - p), 1e-12) / n)
    return abs(p_hat - p) <= z * se


def get_split_stats(df: pd.DataFrame) -> str:
    """
    Generate statistics string for a split DataFrame.

    Args:
        df: DataFrame with 'has_ship' column

    Returns:
        String with counts and ratios information
    """
    vc = df["has_ship"].value_counts()
    frac = (vc / len(df)).round(4).to_dict()
    return f"counts={vc.to_dict()} ratios={frac}"


def validate_inputs(args) -> tuple[Path, Path]:
    """
    Validate input arguments and create output directory.

    Args:
        args: Parsed command line arguments

    Returns:
        Tuple of (csv_path, outdir)

    Raises:
        ValueError: If validation fails
        FileNotFoundError: If input file doesn't exist
    """
    if args.train_name == args.val_name:
        raise ValueError("--train-name and --val-name must be different")

    if not (0.0 < args.val_size < 1.0):
        raise ValueError("--val-size must be in (0,1)")

    outdir = Path(args.outdir)

    # Validate / create output directory
    if outdir.exists():
        if not outdir.is_dir():
            raise ValueError(
                f"Output path exists but is not a directory: {outdir}")
        if not os.access(outdir, os.W_OK):
            raise ValueError(
                f"No write permission for output directory: {outdir}")
    else:
        try:
            outdir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            raise ValueError(
                f"No permission to create output directory: {outdir}")
        except Exception as e:
            raise ValueError(
                f"Failed to create output directory {outdir}: {e}")

    if args.verbose:
        print(f"Output directory: {outdir}")

    # Validate input file existence
    csv_path = Path(args.labels_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV file not found: {csv_path}")
    if not csv_path.is_file():
        raise ValueError(f"Path exists but is not a file: {csv_path}")
    if csv_path.stat().st_size == 0:
        raise ValueError(f"Input CSV file is empty: {csv_path}")

    if args.verbose:
        print(
            f"Reading CSV file: {csv_path} (size: {csv_path.stat().st_size:,} bytes)")

    return csv_path, outdir


def load_and_validate_data(csv_path: Path, verbose: bool = False) -> pd.DataFrame:
    """
    Load and validate the input CSV data.

    Args:
        csv_path: Path to the input CSV file
        verbose: Whether to print verbose output

    Returns:
        Validated DataFrame

    Raises:
        ValueError: If data validation fails
    """
    # Read with nullable dtypes and enforce required columns
    try:
        df = pd.read_csv(
            csv_path,
            dtype={"ImageId": "string", "has_ship": "Int8"},
            usecols=["ImageId", "has_ship"],
        )
    except pd.errors.EmptyDataError:
        raise ValueError(f"CSV file is empty or contains no data: {csv_path}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Failed to parse CSV file {csv_path}: {e}")
    except ValueError as e:
        raise ValueError(
            f"Failed to read required columns from {csv_path}: {e}")
    except Exception as e:
        raise ValueError(f"Unexpected error reading CSV file {csv_path}: {e}")

    if len(df) == 0:
        raise ValueError(f"CSV file contains no rows: {csv_path}")

    # Validate ImageId BEFORE any coercion that could mask nulls
    null_image_ids = df["ImageId"].isna().sum()
    if null_image_ids:
        raise ValueError(
            f"Found {int(null_image_ids)} rows with null ImageId values")
    blank_mask = df["ImageId"].str.strip().eq("")
    if blank_mask.any():
        n_blank = int(blank_mask.sum())
        raise ValueError(f"Found {n_blank} rows with empty ImageId values")

    # Validate has_ship BEFORE casting to small ints
    if df["has_ship"].isna().any():
        n_null = int(df["has_ship"].isna().sum())
        raise ValueError(f"Found {n_null} rows with null has_ship values")

    invalid_values = df[~df["has_ship"].isin([0, 1])]
    if len(invalid_values) > 0:
        invalid_examples = invalid_values[["ImageId", "has_ship"]].head(5)
        error_msg = "Invalid has_ship values found. Expected only 0 or 1, but found:"
        error_msg += f"\n{invalid_examples.to_string(index=False)}"
        if len(invalid_values) > 5:
            error_msg += f"\n... and {len(invalid_values) - 5} more invalid values"
        raise ValueError(error_msg)

    # Normalize/lock dtypes
    df["ImageId"] = df["ImageId"].astype("string")
    df["has_ship"] = df["has_ship"].astype("int8")

    # Duplicate ImageIds: warn (they'll bucket identically; usually desired)
    duplicates = df[df["ImageId"].duplicated(keep=False)]
    if len(duplicates) > 0:
        print(f"⚠️  Warning: Found {len(duplicates)} duplicate ImageId rows")
        print(f"   Example ImageIds: {duplicates['ImageId'].head(3).tolist()}")
        print("   Note: duplicates hash identically and will stay in the same split.")

    # Basic sanity: ensure both classes exist, if present
    vc = df["has_ship"].value_counts()
    if vc.min() < 1:
        raise ValueError("Each class must have at least 1 sample.")
    if verbose:
        print("Data validation passed:")
        print(f"  - Class counts: {vc.to_dict()}")
        print(f"  - Unique ImageIds: {df['ImageId'].nunique():,}")
        print(f"  - Total rows: {len(df):,}")

    return df


def perform_split(df: pd.DataFrame, val_size: float, salt: str, verbose: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform the stratified split of the dataset.

    Args:
        df: Input DataFrame
        val_size: Validation fraction
        salt: Salt string for deterministic hashing
        verbose: Whether to print verbose output

    Returns:
        Tuple of (train_df, val_df)
    """
    # Compute hash values (deterministic, fast)
    if verbose:
        print(
            f"Computing hash values for {len(df):,} samples using pandas hash...")
    hash_values = compute_hash_values(df["ImageId"], salt)

    # Per-class deterministic partition using hash threshold
    parts_train, parts_val = [], []
    for cls, grp in df.groupby("has_ship", sort=False):
        idx = grp.index
        grp_hash = hash_values.loc[idx]
        mask_val = grp_hash < val_size

        val_grp = grp[mask_val]
        train_grp = grp[~mask_val]

        # Edge-case guard: attempt to ensure at least one sample of each class on both sides if possible
        if len(val_grp) == 0 and len(grp) > 1:
            idx_min = grp_hash.idxmin()
            val_grp = grp.loc[[idx_min]]
            train_grp = grp.drop(index=idx_min)

        if len(train_grp) == 0 and len(grp) > 1:
            idx_max = grp_hash.idxmax()
            train_grp = grp.loc[[idx_max]]
            val_grp = grp.drop(index=idx_max)

        parts_train.append(train_grp)
        parts_val.append(val_grp)

    train_df = pd.concat(parts_train, axis=0, ignore_index=True)
    val_df = pd.concat(parts_val, axis=0, ignore_index=True)

    return train_df, val_df


def write_outputs(train_df: pd.DataFrame, val_df: pd.DataFrame, outdir: Path,
                  train_name: str, val_name: str, sort_output: bool,
                  fail_if_exists: bool) -> tuple[Path, Path]:
    """
    Write the split DataFrames to CSV files.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        outdir: Output directory
        train_name: Training file name
        val_name: Validation file name
        sort_output: Whether to sort by ImageId
        fail_if_exists: Whether to fail if files exist

    Returns:
        Tuple of (train_path, val_path)

    Raises:
        FileExistsError: If files exist and fail_if_exists is True
        ValueError: If write fails
    """
    if sort_output:
        train_df = train_df.sort_values(
            "ImageId", kind="mergesort").reset_index(drop=True)
        val_df = val_df.sort_values(
            "ImageId", kind="mergesort").reset_index(drop=True)

    train_path = outdir / train_name
    val_path = outdir / val_name

    # Output existence policy
    existing = [p for p in [train_path, val_path] if p.exists()]
    if existing:
        if fail_if_exists:
            files = "\n".join(f" - {p}" for p in existing)
            raise FileExistsError(
                f"Output files already exist:\n{files}\nUse different names or remove --fail-if-exists."
            )
        else:
            print("⚠️  Warning: Output files already exist and will be overwritten:")
            for p in existing:
                print(f"   - {p}")

    # Write outputs (only required columns)
    try:
        train_df[["ImageId", "has_ship"]].to_csv(train_path, index=False)
        val_df[["ImageId", "has_ship"]].to_csv(val_path, index=False)
    except PermissionError:
        raise ValueError(f"No permission to write output files in {outdir}")
    except Exception as e:
        raise ValueError(f"Failed to write output files: {e}")

    return train_path, val_path


def print_verification_stats(df: pd.DataFrame, train_df: pd.DataFrame,
                             val_df: pd.DataFrame, train_path: Path, val_path: Path,
                             val_size: float, salt: str) -> None:
    """
    Print stratification verification and summary statistics.

    Args:
        df: Original DataFrame
        train_df: Training DataFrame
        val_df: Validation DataFrame
        train_path: Path to training output file
        val_path: Path to validation output file
        val_size: Requested validation size
        salt: Salt used for hashing
    """
    # Stratification verification (stats-aware)
    original_ratio = float(df["has_ship"].mean())
    train_ratio = float(train_df["has_ship"].mean()) if len(train_df) else 0.0
    val_ratio = float(val_df["has_ship"].mean()) if len(val_df) else 0.0
    observed_val_frac = len(val_df) / max(len(df), 1)

    print("\n=== Stratification Verification ===")
    print(
        f"Original positive ratio: {original_ratio:.6f}\n"
        f"Train positive ratio:    {train_ratio:.6f} "
        f"({'OK' if stat_within_tolerance(len(train_df), train_ratio, original_ratio) else 'OFF'})\n"
        f"Val positive ratio:      {val_ratio:.6f} "
        f"({'OK' if stat_within_tolerance(len(val_df), val_ratio, original_ratio) else 'OFF'})"
    )
    print(
        f"Requested val size:      {val_size:.4f}  |  Observed val size: {observed_val_frac:.4f}")

    train_classes = set(train_df["has_ship"].unique())
    val_classes = set(val_df["has_ship"].unique())
    original_classes = set(df["has_ship"].unique())

    missing_train = original_classes - train_classes
    missing_val = original_classes - val_classes

    if missing_train:
        print(f"⚠️  Train split missing classes: {sorted(missing_train)}")
    if missing_val:
        print(f"⚠️  Val split missing classes: {sorted(missing_val)}")
    if not missing_train and not missing_val:
        print("✅ Both splits contain all classes")

    print(
        f"\nWrote: {train_path} ({len(train_df)} rows)  {get_split_stats(train_df)}")
    print(f"Wrote: {val_path} ({len(val_df)} rows)  {get_split_stats(val_df)}")
    overall = (df["has_ship"].value_counts() / len(df)).round(4).to_dict()
    print(f"Overall class ratios: {overall}")
    if salt:
        print(
            f"Salt used: '{salt}' (change to rotate the split deterministically)")
    print("\n=== Performance Summary ===")
    print(f"Dataset size: {len(df):,} samples")
    print("Hash method: pandas (fast 64-bit hash)")
    print("Memory optimization: int8 dtype for binary labels")
    print("Processing: Vectorized hash computation")


def main():
    args = parse_args()

    # Validate inputs and setup
    csv_path, outdir = validate_inputs(args)

    # Load and validate data
    df = load_and_validate_data(csv_path, args.verbose)

    # Perform the split
    train_df, val_df = perform_split(
        df, args.val_size, args.salt, args.verbose)

    # Write outputs
    train_path, val_path = write_outputs(
        train_df, val_df, outdir, args.train_name, args.val_name,
        args.sort_output, args.fail_if_exists
    )

    # Print verification and summary
    print_verification_stats(df, train_df, val_df,
                             train_path, val_path, args.val_size, args.salt)


if __name__ == "__main__":
    main()
