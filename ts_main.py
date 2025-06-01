#!/usr/bin/env python

import importlib
import json
import sys
from datetime import timedelta
from timeit import default_timer as timer
from evaluators import Evaluator
import pandas as pd
import math
import numpy as np
from thompson_sampling import ThompsonSampler
from ts_logger import get_logger
from ts_utils import create_reagents, build_reaction_map
from disallow_tracker import DisallowTracker
from rdkit import Chem
import os 

def read_input(json_filename: str) -> dict:
    """
    Read input parameters from a json file
    :param json_filename: input json file
    :return: a dictionary with the input parameters
    """
    input_data = None
    with open(json_filename, 'r') as ifs:
        input_data = json.load(ifs)
        module = importlib.import_module("evaluators")
        evaluator_class_name = input_data["evaluator_class_name"]
        class_ = getattr(module, evaluator_class_name)
        evaluator_arg = input_data["evaluator_arg"]
        evaluator = class_(evaluator_arg)
        input_data['evaluator_class'] = evaluator
    return input_data


def parse_input_dict(input_data: dict) -> None:
    """
    Parse the input dictionary and add the necessary information
    :param input_data:
    """
    module = importlib.import_module("evaluators")
    evaluator_class_name = input_data["evaluator_class_name"]
    class_ = getattr(module, evaluator_class_name)
    evaluator_arg = input_data["evaluator_arg"]
    evaluator = class_(evaluator_arg)
    input_data['evaluator_class'] = evaluator


def run_ts(input_dict: dict, hide_progress: bool = False) -> None:
    """
    Perform a Thompson sampling run
    :param hide_progress: hide the progress bar
    :param input_dict: dictionary with input parameters
    """
    # 1) Grab whatever was passed in:
    maybe_eval = input_dict["evaluator_class"]
    if isinstance(maybe_eval, Evaluator):
        evaluator = maybe_eval
    else:
        EvaluatorCls = maybe_eval
        evaluator = EvaluatorCls(input_dict["evaluator_arg"])

    # 2) Read *all* reaction SMARTS from reaction_file, into a map reaction_id -> SMARTS
    reaction_file = input_dict["reaction_file"]
    df = pd.read_csv(reaction_file, sep="\t")
    # Build a dict {reaction_id: SMARTS} for every row of the TSV
    reaction_smarts_map = dict(zip(df["reaction_id"].tolist(), df["Reaction"].tolist()))

    # 3) Pull other params
    num_ts_iterations = input_dict["num_ts_iterations"]
    reagent_file = input_dict["reagent_file"]
    num_warmup_trials = input_dict["num_warmup_trials"]
    results_filename = input_dict.get("results_filename")
    ts_mode = input_dict["ts_mode"]
    log_filename = input_dict.get("log_filename")
    batch_size = input_dict.get("batch_size", 1)

    # 4) Set up logger & sampler
    logger = get_logger(__name__, filename=log_filename)
    if results_filename and os.path.exists(results_filename):
        logger.warning(f"Output CSV file '{results_filename}' already exists. Overwriting previous results.")
        try:
            os.remove(results_filename)
            logger.info(f"Successfully removed existing file: {results_filename}")
        except Exception as e:
            logger.error(f"Failed to remove existing file {results_filename}: {e}")
            raise
    ts = ThompsonSampler(mode=ts_mode, log_filename=log_filename, output_csv=results_filename)
    ts.set_reactions(reaction_smarts_map)
    ts.set_hide_progress(hide_progress)
    ts.set_evaluator(evaluator)

    # 5) Load all reagents
    ts.read_reagents([reagent_file])

    # 6) Warm-up & search
    warmup_results = ts.warm_up(num_warmup_trials=num_warmup_trials, batch_size=batch_size)
    search_results = ts.search(ts_num_iterations=num_ts_iterations, batch_size=batch_size)

    # 7) Logging & save
    total_evals = evaluator.counter
    percent_searched = total_evals / ts.get_num_prods() * 100
    logger.info(f"{total_evals} evaluations | {percent_searched:.3f}% of total")

    # 8) Rebuild DataFrame from warmup + search results for printing top hits
    all_results = []

    # warmup_results format: (smiles, name, score) 
    if warmup_results:
        for smiles, name, score in warmup_results:
            all_results.append([score, smiles, name])  # Reorder to [score, smiles, name]

    # search_results format: [score, smiles, name]
    if search_results:
        all_results.extend(search_results)

    def is_valid_smiles(smiles_str):
        """Check if a SMILES string represents a valid molecule"""
        if not smiles_str or not isinstance(smiles_str, str):
            return False
        try:
            mol = Chem.MolFromSmiles(smiles_str)
            return mol is not None
        except:
            return False

    # Filter out FAIL entries before creating DataFrame
    valid_results = []
    for score, smiles, name in all_results:
        # Check if score is finite and SMILES is a valid molecule
        if isinstance(score, (int, float)) and np.isfinite(score) and is_valid_smiles(smiles):
            valid_results.append([score, smiles, name])

    if valid_results:
        out_df = pd.DataFrame(valid_results, columns=["score", "SMILES", "Name"])
        
        # 9) Print top hits
        if not hide_progress:
            ascending = (ts_mode != "maximize")
            top10 = (
                out_df
                .sort_values("score", ascending=ascending)
                .drop_duplicates(subset="SMILES")
                .head(10)
            )
            print(top10)
    else:
        print("No valid results to display")



def run_10_cycles():
    """ A testing function for the paper
    :return: None
    """
    json_file_name = sys.argv[1]
    input_dict = read_input(json_file_name)
    for i in range(0, 10):
        input_dict['results_filename'] = f"ts_result_{i:03d}.csv"
        run_ts(input_dict, hide_progress=False)


def compare_iterations():
    """ A testing function for the paper
    :return:
    """
    json_file_name = sys.argv[1]
    input_dict = read_input(json_file_name)
    for i in (2, 5, 10, 50, 100):
        num_ts_iterations = i * 1000
        input_dict["num_ts_iterations"] = num_ts_iterations
        input_dict["results_filename"] = f"iteration_test_{i}K.csv"
        run_ts(input_dict)


def main():
    start = timer()
    json_filename = sys.argv[1]
    input_dict = read_input(json_filename)
    run_ts(input_dict)
    end = timer()
    print("Elapsed time", timedelta(seconds=end - start))


if __name__ == "__main__":
    main()
