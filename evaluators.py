from abc import ABC, abstractmethod
import numpy as np
import warnings

# silence that “Can’t initialize NVML” message
warnings.filterwarnings(
    "ignore",
    message="Can't initialize NVML",
    category=UserWarning,
    module="torch.cuda"
)

import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from minimol import Minimol
from minimol_dataset import standardize
from minimol_predict import predict_on_precomputed
from rdkit.Chem import Descriptors, AllChem
import os
from contextlib import redirect_stdout, redirect_stderr

class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, mol):
        pass

    @property
    @abstractmethod
    def counter(self):
        pass

class MiniMolEvaluator(Evaluator):
    def __init__(self, config: dict):
        """
        Optimized for Thompson Sampling:
        - Model caching (load once, reuse)
        - GPU utilization
        - Direct model inference
        - No feature caching (Thompson Sampling never revisits molecules)
        """
        self.checkpoints = config["checkpoints"]
        self.tasks = config.get("task_assignments", None)
        if self.tasks is not None and len(self.tasks) != len(self.checkpoints):
            raise ValueError(
                f"task_assignments length ({len(self.tasks)}) "
                f"must equal number of checkpoints ({len(self.checkpoints)})"
            )
        self.mode = config.get("mode", "stl")
        self.architecture = config.get("architecture", "standard")
        self.aggregate = config.get("aggregate", "mean")
        if self.aggregate not in {"mean", "sum", "max"}:
            raise ValueError(
                f"Unsupported aggregate: {self.aggregate}. Use 'mean', 'sum', or 'max'."
            )
        self.featurization_batch_size = config.get("featurization_batch_size", 1024)
        self.log_transform = config.get("log_transform", False)
        self.num_evaluations = 0

        # GPU optimization
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"MiniMolEvaluator using device: {self.device}")

        # Model caching - load once, reuse forever
        self._models_cached = False
        self._loaded_models = []

        # Featurizer (stays on CPU - MiniMol handles this)
        self.featurizer = Minimol(batch_size=self.featurization_batch_size)

    def _load_models_once(self):
        """Load all models once and cache them on GPU"""
        if self._models_cached:
            return
            
        print(f"Loading {len(self.checkpoints)} models to {self.device}...")
        from minimol_predict import load_model
        
        self._loaded_models = []
        task_list = self.tasks if self.tasks is not None else [None] * len(self.checkpoints)
        
        for i, (ckpt, task) in enumerate(zip(self.checkpoints, task_list)):
            print(f"  Loading model {i+1}/{len(self.checkpoints)}: {os.path.basename(ckpt)}")
            model = load_model(ckpt, mode=self.mode, device=str(self.device), 
                             architecture=self.architecture)
            
            # Ensure model is in eval mode and on correct device
            model.eval()
            model.to(self.device)
            
            self._loaded_models.append((model, task))
        
        print(f"All models loaded and cached on {self.device}")
        self._models_cached = True

    @property
    def counter(self) -> int:
        return self.num_evaluations

    def evaluate(self, mol) -> float:
        return self.evaluate_batch([mol])[0]

    def evaluate_batch(self, mols: list) -> list[float]:
        n = len(mols)
        
        # Load models once (only happens on first call)
        self._load_models_once()
        
        # SMILES standardization
        raw = [Chem.MolToSmiles(m) for m in mols]
        std = [standardize(s) for s in raw]
        valid = [i for i, s in enumerate(std) if s is not None]
        if not valid:
            self.num_evaluations += n
            return [float("nan")] * n
        smiles_list = [std[i] for i in valid]

        # Featurization (suppress output)
        with open(os.devnull, 'w') as fnull, redirect_stdout(fnull), redirect_stderr(fnull):
            try:
                feats = self.featurizer(smiles_list)
            except Exception:
                feats, good = [], []
                for idx, smi in zip(valid, smiles_list):
                    try:
                        feats.append(self.featurizer([smi])[0])
                        good.append(idx)
                    except Exception:
                        pass
                valid = good
                if not feats:
                    self.num_evaluations += n
                    return [float("nan")] * n

        # Convert features to tensor and move to GPU
        X = torch.stack([f.float() for f in feats]).to(self.device)

        # Direct model inference (bypassing predict_on_precomputed overhead)
        all_preds = []
        
        for model, task in self._loaded_models:
            with torch.no_grad():
                # Direct model call
                output = model(X)
                probs = torch.sigmoid(output)
                
                # Handle MTL task extraction
                if self.mode == 'mtl' and task is not None:
                    if probs.dim() > 1 and task < probs.shape[1]:
                        probs = probs[:, task]
                    elif probs.dim() > 1:
                        # Fallback: take first task if specified task doesn't exist
                        probs = probs[:, 0]
                    # else: single output, use as-is
                
                # Convert to numpy and flatten
                pred_values = probs.cpu().numpy().flatten()
                all_preds.append(pred_values)

        # Aggregate across models
        M = np.array(all_preds)  # shape (num_models, num_valid_samples)
        if self.aggregate == "mean":
            agg = M.mean(axis=0)
        elif self.aggregate == "sum":
            agg = M.sum(axis=0)
        else:  # max
            agg = M.max(axis=0)

        # Scatter back into full output
        out = [float("nan")] * n
        for j, v in zip(valid, agg):
            out[j] = float(v)

        # Apply log transformation if enabled
        if self.log_transform:
            out = [np.log1p(score) if np.isfinite(score) else score for score in out]

        self.num_evaluations += n
        return out

    def get_memory_info(self):
        """Get current GPU memory usage"""
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3    # GB
            return f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
        return "CPU device - no GPU memory info"










class MWEvaluator(Evaluator):
    """Calculate molecular weight using RDKit's Descriptors."""

    def __init__(self):
        self.num_evaluations = 0

    @property
    def counter(self):
        return self.num_evaluations

    def evaluate(self, mol):
        self.num_evaluations += 1
        # uses RDKit’s built-in Descriptors.MolWt
        return Descriptors.MolWt(mol)

class FPEvaluator(Evaluator):
    """Calculate Tanimoto similarity of Morgan fingerprints to a reference."""

    def __init__(self, input_dict):
        self.ref_smiles = input_dict["query_smiles"]
        self.ref_mol    = Chem.MolFromSmiles(self.ref_smiles)
        # radius=2, nBits=2048 (RDKit defaults)
        self.ref_fp     = AllChem.GetMorganFingerprintAsBitVect(self.ref_mol, radius=2)
        self.num_evaluations = 0

    @property
    def counter(self):
        return self.num_evaluations

    def evaluate(self, mol):
        self.num_evaluations += 1
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2)
        return DataStructs.TanimotoSimilarity(self.ref_fp, fp)


