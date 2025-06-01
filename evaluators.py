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
        config keys:
          - checkpoints: List[str]
          - task_assignments: Optional[List[int]]  # length == len(checkpoints)
          - mode: "stl" or "mtl"
          - architecture: "standard" or "residual"
          - aggregate: "mean", "sum", or "max"
          - featurization_batch_size: int
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
        self.num_evaluations = 0

        # in-memory featurizer
        self.featurizer = Minimol(batch_size=self.featurization_batch_size)

    @property
    def counter(self) -> int:
        return self.num_evaluations

    def evaluate(self, mol) -> float:
        return self.evaluate_batch([mol])[0]

    def evaluate_batch(self, mols: list) -> list[float]:
        n = len(mols)
        # SMILES standardization
        raw = [Chem.MolToSmiles(m) for m in mols]
        std = [standardize(s) for s in raw]
        valid = [i for i, s in enumerate(std) if s is not None]
        if not valid:
            self.num_evaluations += n
            return [float("nan")] * n
        smiles_list = [std[i] for i in valid]

        # Featurization (bulk, with fallback)
        # suppress any tqdm output from Minimol
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

        X = torch.stack([f.float() for f in feats])

        # Predictions per checkpoint + task
        all_preds = []
        task_list = self.tasks if self.tasks is not None else [None] * len(self.checkpoints)
        for ckpt, task in zip(self.checkpoints, task_list):
            df = predict_on_precomputed(
                X_feat=X,
                checkpoints=[ckpt],
                mode=self.mode,
                species_indices=None,
                device=None,
                include_individual_models=False,
                architecture=self.architecture,
                task_of_interest=task,
            )
            mean_cols = [c for c in df.columns if c.endswith("_mean")]
            if not mean_cols:
                raise RuntimeError(f"No _mean column in prediction for {ckpt}")
            col = mean_cols[0]
            all_preds.append(df[col].to_list())

        # Aggregate across checkpoints
        M = np.array(all_preds)  # shape (C, V)
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

        self.num_evaluations += n
        return out










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


