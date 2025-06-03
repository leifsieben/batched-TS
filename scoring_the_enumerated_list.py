#!/usr/bin/env python
"""
corrected_ground_truth_scoring.py

Computes ground truth scores that match the evaluator logic:
- Sum across all 12 model predictions (not mean)
- Both raw scores and normalized scores [0,1]
- Processes data in chunks to avoid memory issues
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import sys
from typing import List, Optional
import gc
import math
torch.cuda.empty_cache()
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import os
os.environ['OMP_NUM_THREADS'] = '1'

# Add MiniMol to path
sys.path.append("/home/lsieben/Thompson_Sampling/MiniMol")
from minimol_dataset import precompute_features          
from minimol_predict import predict_on_precomputed  # Use public API           

# Task assignment: A-B-K-P order, 3 reps each
TASK_INDEX_PER_CHECKPOINT = [0] * 3 + [1] * 3 + [2] * 3 + [3] * 3

CHECKPOINTS = [
    "/home/lsieben/Thompson_Sampling/checkpoints/MTL_by_Andrew_250427/AB_AM_MTL_rep_1_col_1_is_pred.pt",
    "/home/lsieben/Thompson_Sampling/checkpoints/MTL_by_Andrew_250427/AB_AM_MTL_rep_2_col_1_is_pred.pt",
    "/home/lsieben/Thompson_Sampling/checkpoints/MTL_by_Andrew_250427/AB_AM_MTL_rep_3_col_1_is_pred.pt",
    "/home/lsieben/Thompson_Sampling/checkpoints/MTL_by_Andrew_250427/EC_AM_MTL_rep_1_col_2_is_pred.pt",
    "/home/lsieben/Thompson_Sampling/checkpoints/MTL_by_Andrew_250427/EC_AM_MTL_rep_2_col_2_is_pred.pt",
    "/home/lsieben/Thompson_Sampling/checkpoints/MTL_by_Andrew_250427/EC_AM_MTL_rep_3_col_2_is_pred.pt",
    "/home/lsieben/Thompson_Sampling/checkpoints/MTL_by_Andrew_250427/KP_AM_MTL_rep_1_col_3_is_pred.pt",
    "/home/lsieben/Thompson_Sampling/checkpoints/MTL_by_Andrew_250427/KP_AM_MTL_rep_2_col_3_is_pred.pt",
    "/home/lsieben/Thompson_Sampling/checkpoints/MTL_by_Andrew_250427/KP_AM_MTL_rep_3_col_3_is_pred.pt",
    "/home/lsieben/Thompson_Sampling/checkpoints/MTL_by_Andrew_250427/PA_AM_MTL_rep_1_col_4_is_pred.pt",
    "/home/lsieben/Thompson_Sampling/checkpoints/MTL_by_Andrew_250427/PA_AM_MTL_rep_2_col_4_is_pred.pt",
    "/home/lsieben/Thompson_Sampling/checkpoints/MTL_by_Andrew_250427/PA_AM_MTL_rep_3_col_4_is_pred.pt",
]

def get_tensor_info(tensor_path: str) -> tuple:
    """Get tensor shape without loading it fully into memory."""
    # Load just the tensor metadata
    with open(tensor_path, 'rb') as f:
        # PyTorch tensors store shape info in the header
        tensor = torch.load(f, map_location='cpu')
        shape = tensor.shape
        dtype = tensor.dtype
        del tensor
        torch.cuda.empty_cache()
        gc.collect()
        return shape, dtype

def load_tensor_chunk(tensor_path: str, start_idx: int, end_idx: int) -> torch.Tensor:
    """Load a specific chunk of a tensor."""
    full_tensor = torch.load(tensor_path, map_location='cpu')
    chunk = full_tensor[start_idx:end_idx].clone()
    del full_tensor
    torch.cuda.empty_cache()
    gc.collect()
    return chunk

def compute_ground_truth_scores(input_csv: str, output_csv: str, 
                               precomputed_features_path: str = None,
                               metadata_csv_path: str = None,
                               chunk_size: int = 50000) -> None:
    """
    Compute ground truth scores exactly as the evaluator does:
    1. Get prediction from each of the 12 checkpoints for their respective tasks
    2. Sum all 12 predictions (no averaging)
    3. Normalize to [0,1] using theoretical range [0,12]
    4. Process in chunks to manage memory usage
    """
    
    print(f"Computing ground truth scores with chunk size: {chunk_size}...")
    
    # ------------------------------------------------------------------ STEP 1
    # Get tensor info or compute features
    if precomputed_features_path and os.path.exists(precomputed_features_path):
        print(f"Getting tensor info from {precomputed_features_path}")
        tensor_shape, tensor_dtype = get_tensor_info(precomputed_features_path)
        features_path = precomputed_features_path
    else:
        print("Computing features...")
        precomp_dir = os.path.splitext(output_csv)[0] + "_precomp"
        features_path, meta_csv = precompute_features(
            input_csv,
            output_dir=precomp_dir,
            smiles_col="SMILES",
            standardize_smiles=True,
            batch_size=10_000,
            skip_if_exists=True,
        )
        tensor_shape, tensor_dtype = get_tensor_info(features_path)
        metadata_csv_path = meta_csv
    
    n_samples = tensor_shape[0]
    n_chunks = math.ceil(n_samples / chunk_size)
    print(f"Processing {n_samples} molecules in {n_chunks} chunks of max {chunk_size} each...")
    
    # ------------------------------------------------------------------ STEP 2
    # Initialize arrays to store results
    all_predictions = np.zeros((len(CHECKPOINTS), n_samples), dtype=np.float32)
    
    # Process each chunk
    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, n_samples)
        current_chunk_size = end_idx - start_idx
        
        print(f"\n--- Processing chunk {chunk_idx + 1}/{n_chunks} (samples {start_idx}:{end_idx}) ---")
        
        # Load chunk of features
        X_chunk = load_tensor_chunk(features_path, start_idx, end_idx)
        print(f"Loaded chunk shape: {X_chunk.shape}")
        
        # Get predictions from all 12 checkpoints for this chunk
        for i, ckpt in enumerate(CHECKPOINTS):
            task_idx = TASK_INDEX_PER_CHECKPOINT[i]
            task_names = ["AB", "EC", "KP", "PA"]
            
            print(f"  Checkpoint {i+1}/12: {task_names[task_idx]} task (task {task_idx})")
            
            # Get prediction for this specific task using the public API
            df_pred = predict_on_precomputed(
                X_feat=X_chunk,
                checkpoints=[ckpt],
                mode="mtl",
                species_indices=None,
                device='cuda',
                include_individual_models=False,
                architecture="residual",
                task_of_interest=task_idx,
            )
            
            # Extract the mean prediction for this task
            mean_cols = [c for c in df_pred.columns if c.endswith("_mean")]
            if not mean_cols:
                raise RuntimeError(f"No _mean column found for checkpoint {ckpt}")
            
            if len(mean_cols) != 1:
                raise RuntimeError(f"Expected 1 mean column, got {len(mean_cols)}: {mean_cols}")
            
            # Store predictions for this chunk
            all_predictions[i, start_idx:end_idx] = df_pred[mean_cols[0]].values
            
            # Clear GPU memory
            torch.cuda.empty_cache()
        
        # Clear chunk from memory
        del X_chunk
        torch.cuda.empty_cache()
        gc.collect()
        
        # Print progress
        pred_stats = all_predictions[:, start_idx:end_idx].sum(axis=0)
        print(f"  Chunk raw scores: min={pred_stats.min():.4f}, max={pred_stats.max():.4f}, "
              f"mean={pred_stats.mean():.4f}")
    
    print("\n--- All chunks processed, computing final statistics ---")
    
    # ------------------------------------------------------------------ STEP 3
    # Compute raw scores (sum across all 12 models) - this matches the evaluator
    raw_scores = all_predictions.sum(axis=0)  # (N,)
    
    # Compute normalized scores using theoretical range [0, 12]
    theoretical_min = 0.0
    theoretical_max = 12.0
    normalized_scores = (raw_scores - theoretical_min) / (theoretical_max - theoretical_min)
    # Clip to [0,1] in case models go outside expected range
    normalized_scores = np.clip(normalized_scores, 0.0, 1.0)
    
    # ------------------------------------------------------------------ STEP 4
    # Compute individual task means for analysis
    ab_mean = all_predictions[0:3].mean(axis=0)
    ec_mean = all_predictions[3:6].mean(axis=0)
    kp_mean = all_predictions[6:9].mean(axis=0)
    pa_mean = all_predictions[9:12].mean(axis=0)
    
    # ------------------------------------------------------------------ STEP 5
    # Create output DataFrame in chunks to manage memory
    print("Creating output DataFrame...")
    
    if metadata_csv_path and os.path.exists(metadata_csv_path):
        # Process metadata CSV in chunks too if it's large
        print(f"Loading metadata from {metadata_csv_path}")
        try:
            metadata_df = pd.read_csv(metadata_csv_path)
            if len(metadata_df) != n_samples:
                print(f"WARNING: Metadata has {len(metadata_df)} rows but expected {n_samples}")
        except MemoryError:
            print("Metadata CSV too large, will use minimal DataFrame")
            metadata_df = None
    else:
        metadata_df = None
    
    if metadata_df is not None:
        out_df = metadata_df.copy()
    else:
        # Create minimal DataFrame with SMILES
        print("Creating minimal DataFrame with SMILES...")
        input_df = pd.read_csv(input_csv)
        if len(input_df) != n_samples:
            print(f"WARNING: Input CSV has {len(input_df)} rows but expected {n_samples}")
        out_df = input_df[['SMILES']].copy()
    
    # Add scores
    print("Adding score columns...")
    out_df["raw_score"] = raw_scores
    out_df["normalized_score"] = normalized_scores
    
    # Add individual task means for debugging/analysis
    out_df["AB_mean"] = ab_mean
    out_df["EC_mean"] = ec_mean  
    out_df["KP_mean"] = kp_mean
    out_df["PA_mean"] = pa_mean
    
    # Add individual model predictions for maximum debugging
    for i in range(12):
        task_idx = TASK_INDEX_PER_CHECKPOINT[i]
        task_names = ["AB", "EC", "KP", "PA"]
        rep_num = (i % 3) + 1
        col_name = f"{task_names[task_idx]}_rep{rep_num}"
        out_df[col_name] = all_predictions[i]
    
    # ------------------------------------------------------------------ STEP 6
    # Report statistics
    print("\n=== SCORING STATISTICS ===")
    print(f"Raw scores:")
    print(f"  Range: {raw_scores.min():.4f} to {raw_scores.max():.4f}")
    print(f"  Mean: {raw_scores.mean():.4f} ± {raw_scores.std():.4f}")
    print(f"  Median: {np.median(raw_scores):.4f}")
    print(f"  95th percentile: {np.percentile(raw_scores, 95):.4f}")
    
    print(f"\nNormalized scores:")
    print(f"  Range: {normalized_scores.min():.4f} to {normalized_scores.max():.4f}")
    print(f"  Mean: {normalized_scores.mean():.4f} ± {normalized_scores.std():.4f}")
    
    # Check if any scores are outside [0,12] range
    outside_range = np.sum((raw_scores < 0) | (raw_scores > 12))
    if outside_range > 0:
        print(f"\nWARNING: {outside_range} molecules have scores outside [0,12] range!")
        print(f"  This suggests the theoretical range assumption may be incorrect.")
    
    # Individual task statistics
    print(f"\nTask-specific means:")
    print(f"  AB: {ab_mean.mean():.4f} ± {ab_mean.std():.4f}")
    print(f"  EC: {ec_mean.mean():.4f} ± {ec_mean.std():.4f}")
    print(f"  KP: {kp_mean.mean():.4f} ± {kp_mean.std():.4f}")
    print(f"  PA: {pa_mean.mean():.4f} ± {pa_mean.std():.4f}")
    
    # ------------------------------------------------------------------ STEP 7
    # Save results
    print(f"Saving results to {output_csv}...")
    out_df.to_csv(output_csv, index=False)
    print(f"\n[✓] Saved ground truth scores → {output_csv}")
    print(f"Columns: {list(out_df.columns)}")
    
    # Clean up
    del all_predictions, raw_scores, normalized_scores
    torch.cuda.empty_cache()
    gc.collect()

def main():
    parser = argparse.ArgumentParser(description="Compute ground truth scores matching evaluator logic")
    parser.add_argument("--input_csv", required=True, help="Input CSV with SMILES column")
    parser.add_argument("--output_csv", required=True, help="Output CSV for scores")
    parser.add_argument("--precomputed_features", help="Path to precomputed features.pt file")
    parser.add_argument("--metadata_csv", help="Path to metadata.csv file")
    parser.add_argument("--chunk_size", type=int, default=50000, 
                       help="Number of samples to process at once (default: 50000)")
    
    args = parser.parse_args()
    
    compute_ground_truth_scores(
        args.input_csv,
        args.output_csv,
        args.precomputed_features,
        args.metadata_csv,
        args.chunk_size
    )

if __name__ == "__main__":
    main()