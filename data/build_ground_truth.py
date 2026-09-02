#!/usr/bin/env python
"""
Build (or rebuild) data/ground_truth.json to match the current batch size.

Usage
-----
    python data/build_ground_truth.py [--records N] [--seed S]

This is a thin wrapper around SyntheticDataGenerator.generate() that ensures
the ground_truth.json on disk covers exactly the same records as the batch
files (bank_statement.csv, internal_ledger.csv, settlements_live.csv).

Run this whenever you change --records or --seed so that strict-mode
coverage stays at 1.0.

The script DOES re-generate bank_statement.csv and internal_ledger.csv
(via generate()) — that is intentional and keeps all three files in sync.
"""

import argparse
import os
import sys

# Make sure the project root is on the path when running directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from data.generate_data import SyntheticDataGenerator


def build_ground_truth(
    num_records: int = 80,
    seed: int = 42,
    messiness: float = 0.25,
    settlements_path: str = "data/settlements_live.csv",
    output_dir: str = "data",
):
    """
    Regenerate bank_statement.csv, internal_ledger.csv, and ground_truth.json
    as a coherent, fully-labeled batch.

    Args:
        num_records:      Number of records to generate.
        seed:             Random seed (same seed = same output, proving determinism).
        messiness:        Fraction of records with injected issues (0.0–1.0).
        settlements_path: If a settlements CSV exists here it will be used to
                          link some records to real Razorpay settlement IDs.
        output_dir:       Directory to write output files (default: data/).
    """
    # Optionally link to real settlements
    settlements_df = None
    if os.path.exists(settlements_path):
        settlements_df = pd.read_csv(settlements_path)
        print(f"Loaded {len(settlements_df)} settlements from {settlements_path}")
    else:
        print(f"No settlements file at {settlements_path} — generating fully synthetic data.")

    generator = SyntheticDataGenerator(seed=seed, messiness_ratio=messiness)
    _, _, ground_truth = generator.generate(
        num_records=num_records,
        settlements_df=settlements_df,
        output_dir=output_dir,
    )

    gt_path = os.path.join(output_dir, "ground_truth.json")
    print(f"\n[OK] ground_truth.json written with {len(ground_truth)} entries -> {gt_path}")
    print(f"     Every record has a labeled 'should_match' field.")
    print(f"     Re-run the pipeline now to get ground_truth_coverage = 1.0")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild data/ground_truth.json so it covers the full batch. "
            "Always use the same --records and --seed as your pipeline run."
        )
    )
    parser.add_argument("--records", type=int, default=80,
                        help="Number of records to generate (default: 80)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--messiness", type=float, default=0.25,
                        help="Fraction of records with injected issues (default: 0.25)")
    parser.add_argument("--settlements", type=str, default="data/settlements_live.csv",
                        help="Path to settlements CSV for linking (optional)")
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Output directory for generated files (default: data/)")
    args = parser.parse_args()

    build_ground_truth(
        num_records=args.records,
        seed=args.seed,
        messiness=args.messiness,
        settlements_path=args.settlements,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
