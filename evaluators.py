import os
import warnings
from abc import ABC, abstractmethod
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import numpy as np
import torch
from rdkit import Chem
from minimol import Minimol
from minimol_dataset import standardize
from minimol_predict import predict_on_precomputed
import useful_rdkit_utils as uru

try:
    from openeye import oechem
    from openeye import oeomega
    from openeye import oeshape
    from openeye import oedocking
    import joblib
except ImportError:
    # Since openeye is a commercial software package, just pass with a warning if not available
    warnings.warn(f"Openeye packages not available in this environment; do not attempt to use ROCSEvaluator or "
                  f"FredEvaluator")
from rdkit import Chem, DataStructs
import pandas as pd
from sqlitedict import SqliteDict

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
    """A simple evaluation class that calculates molecular weight, this was just a development tool
    """

    def __init__(self):
        self.num_evaluations = 0

    @property
    def counter(self):
        return self.num_evaluations

    def evaluate(self, mol):
        self.num_evaluations += 1
        return uru.MolWt(mol)


class FPEvaluator(Evaluator):
    """An evaluator class that calculates a fingerprint Tanimoto to a reference molecule
    """

    def __init__(self, input_dict):
        self.ref_smiles = input_dict["query_smiles"]
        self.fpgen = rdFingerprintGenerator.GetMorganGenerator()
        self.ref_mol = Chem.MolFromSmiles(self.ref_smiles)
        self.ref_fp = self.fpgen.GetFingerprint(self.ref_mol)
        self.num_evaluations = 0
        self.fpgen = rdFingerprintGenerator.GetMorganGenerator()

    @property
    def counter(self):
        return self.num_evaluations

    def evaluate(self, rd_mol_in):
        self.num_evaluations += 1
        rd_mol_fp = self.fpgen.GetFingerprint(rd_mol_in)
        return DataStructs.TanimotoSimilarity(self.ref_fp, rd_mol_fp)


class ROCSEvaluator(Evaluator):
    """An evaluator class that calculates a ROCS score to a reference molecule
    """

    def __init__(self, input_dict):
        ref_filename = input_dict['query_molfile']
        ref_fs = oechem.oemolistream(ref_filename)
        self.ref_mol = oechem.OEMol()
        oechem.OEReadMolecule(ref_fs, self.ref_mol)
        self.max_confs = 50
        self.score_cache = {}
        self.num_evaluations = 0

    @property
    def counter(self):
        return self.num_evaluations

    def set_max_confs(self, max_confs):
        """Set the maximum number of conformers generated by Omega
        :param max_confs:
        """
        self.max_confs = max_confs

    def evaluate(self, rd_mol_in):
        """Generate conformers with Omega and evaluate the ROCS overlay of conformers to a reference molecule
        :param rd_mol_in: Input RDKit molecule
        :return: ROCS Tanimoto Combo score, returns -1 if conformer generation fails
        """
        self.num_evaluations += 1
        smi = Chem.MolToSmiles(rd_mol_in)
        # Look up to see if we already processed this molecule
        arc_tc = self.score_cache.get(smi)
        if arc_tc is not None:
            tc = arc_tc
        else:
            fit_mol = oechem.OEMol()
            oechem.OEParseSmiles(fit_mol, smi)
            ret_code = generate_confs(fit_mol, self.max_confs)
            if ret_code:
                tc = self.overlay(fit_mol)
            else:
                tc = -1.0
            self.score_cache[smi] = tc
        return tc

    def overlay(self, fit_mol):
        """Use ROCS to overlay two molecules
        :param fit_mol: OEMolecule
        :return: Combo Tanimoto for the overlay
        """
        prep = oeshape.OEOverlapPrep()
        prep.Prep(self.ref_mol)
        overlay = oeshape.OEMultiRefOverlay()
        overlay.SetupRef(self.ref_mol)
        prep.Prep(fit_mol)
        score = oeshape.OEBestOverlayScore()
        overlay.BestOverlay(score, fit_mol, oeshape.OEHighestTanimoto())
        return score.GetTanimotoCombo()


class LookupEvaluator(Evaluator):
    """A simple evaluation class that looks up values from a file.
    This is primarily used for testing.
    """

    def __init__(self, input_dictionary):
        self.num_evaluations = 0
        ref_filename = input_dictionary['ref_filename']
        ref_colname = input_dictionary['ref_colname']
        if ref_filename.endswith(".csv"):
            ref_df = pd.read_csv(ref_filename)
        elif ref_filename.endswith(".parquet"):
            ref_df = pd.read_parquet(ref_filename)
        else:
            print(ref_filename,"does not have valid extendsion must be in [.csv,.parquet]")
            assert(False)
        self.ref_dict = dict([(a, b) for a, b in ref_df[['SMILES', ref_colname]].values])

    @property
    def counter(self):
        return self.num_evaluations

    def evaluate(self, mol):
        self.num_evaluations += 1
        smi = Chem.MolToSmiles(mol)
        val = self.ref_dict.get(smi)
        if val is not None:
            return val
        else:
            return np.nan

class DBEvaluator(Evaluator):
    """A simple evaluator class that looks up values from a database.
    This is primarily used for benchmarking
    """

    def __init__(self, input_dictionary):
        self.num_evaluations = 0
        self.db_prefix = input_dictionary['db_prefix']
        db_filename = input_dictionary['db_filename']
        self.ref_dict = SqliteDict(db_filename)

    def __repr__(self):
        return "DBEvalutor"


    @property
    def counter(self):
        return self.num_evaluations


    def evaluate(self, smiles):
        self.num_evaluations += 1
        res = self.ref_dict.get(f"{self.db_prefix}{smiles}")
        if res is None:
            return np.nan
        else:
            if res == -500:
                return np.nan
            return res
    

class FredEvaluator(Evaluator):
    """An evaluator class that docks a molecule with the OEDocking Toolkit and returns the score
    """

    def __init__(self, input_dict):
        du_file = input_dict["design_unit_file"]
        if not os.path.isfile(du_file):
            raise FileNotFoundError(f"{du_file} was not found or is a directory")
        self.dock = read_design_unit(du_file)
        self.num_evaluations = 0
        self.max_confs = 50

    @property
    def counter(self):
        return self.num_evaluations

    def set_max_confs(self, max_confs):
        """Set the maximum number of conformers generated by Omega
        :param max_confs:
        """
        self.max_confs = max_confs

    def evaluate(self, mol):
        self.num_evaluations += 1
        smi = Chem.MolToSmiles(mol)
        mc_mol = oechem.OEMol()
        oechem.OEParseSmiles(mc_mol, smi)
        confs_ok = generate_confs(mc_mol, self.max_confs)
        score = 1000.0
        docked_mol = oechem.OEGraphMol()
        if confs_ok:
            ret_code = self.dock.DockMultiConformerMolecule(docked_mol, mc_mol)
        else:
            ret_code = oedocking.OEDockingReturnCode_ConformerGenError
        if ret_code == oedocking.OEDockingReturnCode_Success:
            dock_opts = oedocking.OEDockOptions()
            sd_tag = oedocking.OEDockMethodGetName(dock_opts.GetScoreMethod())
            # this is a stupid hack, I need to figure out how to do this correctly
            oedocking.OESetSDScore(docked_mol, self.dock, sd_tag)
            score = float(oechem.OEGetSDData(docked_mol, sd_tag))
        return score


def generate_confs(mol, max_confs):
    """Generate conformers with Omega
    :param max_confs: maximum number of conformers to generate
    :param mol: input OEMolecule
    :return: Boolean Omega return code indicating success of conformer generation
    """
    rms = 0.5
    strict_stereo = False
    omega = oeomega.OEOmega()
    omega.SetRMSThreshold(rms)  # Word to the wise: skipping this step can lead to significantly different charges!
    omega.SetStrictStereo(strict_stereo)
    omega.SetMaxConfs(max_confs)
    error_level = oechem.OEThrow.GetLevel()
    # Turn off OEChem warnings
    oechem.OEThrow.SetLevel(oechem.OEErrorLevel_Error)
    status = omega(mol)
    # Turn OEChem warnings back on
    oechem.OEThrow.SetLevel(error_level)
    return status


def read_design_unit(filename):
    """Read an OpenEye design unit
    :param filename: design unit filename (.oedu)
    :return: a docking grid
    """
    du = oechem.OEDesignUnit()
    rfs = oechem.oeifstream()
    if not rfs.open(filename):
        oechem.OEThrow.Fatal("Unable to open %s for reading" % filename)

    du = oechem.OEDesignUnit()
    if not oechem.OEReadDesignUnit(rfs, du):
        oechem.OEThrow.Fatal("Failed to read design unit")
    if not du.HasReceptor():
        oechem.OEThrow.Fatal("Design unit %s does not contain a receptor" % du.GetTitle())
    dock_opts = oedocking.OEDockOptions()
    dock = oedocking.OEDock(dock_opts)
    dock.Initialize(du)
    return dock


def test_fred_eval():
    """Test function for the Fred docking Evaluator
    :return: None
    """
    input_dict = {"design_unit_file": "data/2zdt_receptor.oedu"}
    fred_eval = FredEvaluator(input_dict)
    smi = "CCSc1ncc2c(=O)n(-c3c(C)nc4ccccn34)c(-c3[nH]nc(C)c3F)nc2n1"
    mol = Chem.MolFromSmiles(smi)
    score = fred_eval.evaluate(mol)
    print(score)


def test_rocs_eval():
    """Test function for the ROCS evaluator
    :return: None
    """
    input_dict = {"query_molfile": "data/2chw_lig.sdf"}
    rocs_eval = ROCSEvaluator(input_dict)
    smi = "CCSc1ncc2c(=O)n(-c3c(C)nc4ccccn34)c(-c3[nH]nc(C)c3F)nc2n1"
    mol = Chem.MolFromSmiles(smi)
    combo_score = rocs_eval.evaluate(mol)
    print(combo_score)


class MLClassifierEvaluator(Evaluator):
    """An evaluator class the calculates a score based on a trained ML model
    """

    def __init__(self, input_dict):
        self.cls = joblib.load(input_dict["model_filename"])
        self.num_evaluations = 0

    @property
    def counter(self):
        return self.num_evaluations

    def evaluate(self, mol):
        self.num_evaluations += 1
        fp = uru.mol2morgan_fp(mol)
        return self.cls.predict_proba([fp])[:,1][0]


def test_ml_classifier_eval():
    """Test function for the ML Classifier Evaluator
    :return: None
    """
    input_dict = {"model_filename": "mapk1_modl.pkl"}
    ml_cls_eval = MLClassifierEvaluator(input_dict)
    smi = "CCSc1ncc2c(=O)n(-c3c(C)nc4ccccn34)c(-c3[nH]nc(C)c3F)nc2n1"
    mol = Chem.MolFromSmiles(smi)
    score = ml_cls_eval.evaluate(mol)
    print(score)


if __name__ == "__main__":
    test_rocs_eval()
