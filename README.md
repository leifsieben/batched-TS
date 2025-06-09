# Thompson Sampling for Virtual Screening

Thompson Sampling is an active learning strategy for efficiently searching large, un-enumerated chemical libraries. This implementation uses a reagent-first sampling approach where the algorithm first selects the most promising reagent across all reactions using Thompson Sampling, then completes the molecule by filling remaining reaction slots while avoiding duplicates. This repo is forked from previous work published here ["Thompson Sampling─An Efficient Method for Searching Ultralarge Synthesis on Demand Databases"](https://pubs.acs.org/doi/10.1021/acs.jcim.3c01790).

## Installation

```bash
conda create -c conda-forge -n thompson-sampling rdkit
conda activate thompson-sampling
pip install -r requirements.txt
```

## Usage 

```python 
python ts_main.py config.json
```

## Configuration Parameters

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `evaluator_class_name` | string | Name of evaluator class: "MiniMolEvaluator", "FPEvaluator", "MWEvaluator" |
| `evaluator_arg` | dict | Arguments passed to evaluator constructor (see Evaluator Parameters below) |
| `reaction_file` | string | Path to TSV file with columns: reaction_id, Reaction |
| `reagent_file` | string | Path to TSV file with columns: SMILES, synton_id, synton#, reaction_id, release |
| `num_ts_iterations` | int | Number of Thompson Sampling iterations (typically 100-2000) |
| `num_warmup_trials` | int | Random samples per reagent during warmup (3 for 2-component, 10+ for 3+ components) |
| `ts_mode` | string | Optimization direction: "maximize" or "minimize" |

### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `results_filename` | string | null | Output CSV file path for results |
| `log_filename` | string | null | Log file path (stdout if null) |
| `batch_size` | int | 128 | Molecules per evaluation batch |
| `min_enumeration_attempts` | int | 1000 | Minimum enumeration attempts per batch |

## Evaluator Parameters

### MiniMolEvaluator

High-performance ML evaluator with GPU acceleration and model caching.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `checkpoints` | list[string] | required | Paths to model checkpoint files |
| `task_assignments` | list[int] | null | Task index each model should predict (MTL only) |
| `mode` | string | "stl" | Model type: "stl" (single-task) or "mtl" (multi-task) |
| `architecture` | string | "standard" | Model architecture: "standard" or "residual" |
| `aggregate` | string | "mean" | Ensemble aggregation: "mean", "sum", or "max" |
| `featurization_batch_size` | int | 1024 | Batch size for molecular featurization |
| `log_transform` | bool | false | Apply log(1 + score) transformation to scores |

### FPEvaluator

2D fingerprint similarity using Morgan fingerprints.

| Parameter | Type | Description |
|-----------|------|-------------|
| `query_smiles` | string | SMILES string of query molecule |

### MWEvaluator

Molecular weight calculator (no parameters required).

| Parameter | Type | Description |
|-----------|------|-------------|
| (none) | - | No parameters required |

## Example Configuration 

```python 
{
  "evaluator_class_name": "MiniMolEvaluator",
  "evaluator_arg": {
    "checkpoints": [
      "models/bioactivity_model1.ckpt",
      "models/bioactivity_model2.ckpt",
      "models/bioactivity_model3.ckpt"
    ],
    "mode": "mtl",
    "task_assignments": [0, 0, 1],
    "architecture": "standard",
    "aggregate": "mean",
    "featurization_batch_size": 4096,
    "log_transform": false
  },
  "reaction_file": "data/reactions.tsv",
  "reagent_file": "data/reagents.tsv",
  "num_ts_iterations": 1000,
  "num_warmup_trials": 3,
  "ts_mode": "maximize",
  "batch_size": 10000,
  "results_filename": "bioactivity_results.csv",
  "log_filename": "bioactivity_search.log"
}
```