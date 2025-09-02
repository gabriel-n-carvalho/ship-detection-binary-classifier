#!/usr/bin/env python3
"""
Convert Airbus ship detection segmentation data to binary classification labels.

This script processes the Kaggle Airbus Ship Detection Challenge segmentation CSV
and converts it to binary labels (has_ship: 0/1) for each unique image.
"""

import argparse
import pandas as pd
from pathlib import Path
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Airbus segmentation to binary labels")
    parser.add_argument(
        "--input",
        default="src/airbus-ship-detection/train_ship_segmentations_v2.csv",
        help="Input CSV file path"
    )
    parser.add_argument(
        "--output",
        default="src/airbus_binary_labels.csv",
        help="Output CSV file path"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100000,
        help="Chunk size for large files (use 0 for no chunking)"
    )
    return parser.parse_args()


def validate_csv_structure(df):
    """Validate the CSV structure and data quality."""
    print("Validating CSV structure...")

    # Check required columns
    required_cols = {"ImageId", "EncodedPixels"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Check for empty dataframe
    if df.empty:
        raise ValueError("CSV file is empty")

    # Check for missing ImageIds
    missing_image_ids = df["ImageId"].isna().sum()
    if missing_image_ids > 0:
        print(
            f"⚠️  Warning: Found {missing_image_ids} rows with missing ImageId")

    # Validate EncodedPixels format (basic check)
    valid_pixels = df["EncodedPixels"].dropna()
    if len(valid_pixels) > 0:
        sample = valid_pixels.iloc[0]
        if not isinstance(sample, str):
            print("⚠️  Warning: EncodedPixels may not be in expected string format")

    print("✅ CSV structure validation passed")


def process_chunked_data(csv_path, chunk_size):
    """Process data in chunks to handle large files efficiently."""
    print(f"Processing data in chunks of {chunk_size:,} rows...")

    chunks = []
    total_rows = 0

    for i, chunk in enumerate(pd.read_csv(csv_path, usecols=["ImageId", "EncodedPixels"], chunksize=chunk_size)):
        chunks.append(chunk)
        total_rows += len(chunk)
        print(
            f"  Processed chunk {i+1}: {len(chunk):,} rows (total: {total_rows:,})")

    df = pd.concat(chunks, ignore_index=True)
    print(f"✅ Loaded {len(df):,} total rows from {len(chunks)} chunks")
    return df


def main():
    args = parse_args()

    # Validate input file
    csv_path = Path(args.input)
    if not csv_path.exists():
        print(f"❌ Error: Input file not found: {csv_path}")
        sys.exit(1)

    print(f"Processing: {csv_path}")
    print(f"Output: {args.output}")

    try:
        # Load data (chunked or full)
        if args.chunk_size > 0:
            df = process_chunked_data(csv_path, args.chunk_size)
        else:
            print("Loading entire file into memory...")
            df = pd.read_csv(csv_path, usecols=["ImageId", "EncodedPixels"])
            print(f"✅ Loaded {len(df):,} rows")

        validate_csv_structure(df)

        # Print initial statistics
        print(f"\nInitial data statistics:")
        print(f"  Total segmentation records: {len(df):,}")
        print(f"  Unique images: {df['ImageId'].nunique():,}")
        print(f"  Records with ships: {df['EncodedPixels'].notna().sum():,}")
        print(f"  Records without ships: {df['EncodedPixels'].isna().sum():,}")

        # Convert to binary labels
        print(f"\nConverting to binary labels...")
        has_ship = (
            df
            .assign(has_ship=df["EncodedPixels"].notna().astype(int))
            .groupby("ImageId", as_index=False)["has_ship"].max()
            .sort_values("ImageId")
        )

        # Print final statistics
        print(f"\nFinal binary labels:")
        class_counts = has_ship["has_ship"].value_counts(dropna=False)
        print(f"  Images with ships (1): {class_counts.get(1, 0):,}")
        print(f"  Images without ships (0): {class_counts.get(0, 0):,}")
        print(f"  Class balance: {has_ship['has_ship'].mean():.3f}")

        # Save output
        output_path = Path(args.output)
        has_ship.to_csv(output_path, index=False)
        print(f"\n✅ Saved binary labels to: {output_path}")
        print(f"   File size: {output_path.stat().st_size:,} bytes")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
