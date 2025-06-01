#!/usr/bin/env python
"""
precompute_reagents.py

Reads a tab-separated file of synthons with columns:
SMILES    synton_id    synton#    reaction_id    release

Creates Reagent objects for each synthon and saves them to a pickle file.
"""

import os
import argparse
import joblib
import pandas as pd

from reagent import Reagent  # Reagent class defined here :contentReference[oaicite:0]{index=0}


def load_reagents_from_file(reagent_file: str) -> list[Reagent]:
    """
    Read the synthons file and instantiate a Reagent for each row.
    Expects columns: SMILES, synton_id, synton#, reaction_id, release
    """
    dtype = {
        "SMILES": str,
        "synton_id": str,
        "synton#": int,
        "reaction_id": str,
        "release": str
    }
    df = pd.read_csv(reagent_file, sep="\t", dtype=dtype)

    reagents: list[Reagent] = []
    for _, row in df.iterrows():
        smiles = row["SMILES"]
        reagent_name = row["synton_id"]
        reaction_id = row["reaction_id"]
        synton_idx = int(row["synton#"])
        reagent = Reagent(
            reagent_name=reagent_name,
            smiles=smiles,
            reaction_id=reaction_id,
            synton_idx=synton_idx
        )
        reagents.append(reagent)

    return reagents


def save_reagents(reagents: list[Reagent], output_file: str) -> None:
    """
    Save the list of Reagent objects to a pickle file using joblib.
    Overwrites output_file if it already exists.
    """
    # Ensure the directory exists
    out_dir = os.path.dirname(output_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Remove existing file to avoid appending
    if os.path.exists(output_file):
        os.remove(output_file)

    joblib.dump(reagents, output_file)
    print(f"✅ Saved {len(reagents)} Reagent objects to {output_file}.")


def main():
    parser = argparse.ArgumentParser(
        description="Precompute Reagent objects from a synthons TSV file."
    )
    parser.add_argument(
        "reagent_file",
        type=str,
        help="Path to input TSV file of synthons (columns: SMILES, synton_id, synton#, reaction_id, release)"
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to output pickle file for precomputed Reagent objects"
    )
    args = parser.parse_args()

    print("📥 Loading synthons from:", args.reagent_file)
    reagents = load_reagents_from_file(args.reagent_file)

    print("🔄 Saving precomputed Reagent objects to:", args.output_file)
    save_reagents(reagents, args.output_file)


if __name__ == "__main__":
    main()
