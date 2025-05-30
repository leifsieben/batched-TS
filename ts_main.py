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
    # If it's already an Evaluator instance, use it. Otherwise, treat it as a class.
    if isinstance(maybe_eval, Evaluator):
        evaluator = maybe_eval
    else:
        # assume it's a class you need to instantiate
        EvaluatorCls = maybe_eval
        evaluator    = EvaluatorCls(input_dict["evaluator_arg"])

    # 2) Grab reaction SMARTS from your reaction_file TSV
    df = pd.read_csv(input_dict["reaction_file"], sep="\t")
    reaction_smarts = df.loc[0, "Reaction"]
    reaction_id     = df.loc[0, "reaction_id"] 

    # 3) Pull other params
    num_ts_iterations  = input_dict["num_ts_iterations"]
    reagent_file  = input_dict["reagent_file"]
    num_warmup_trials  = input_dict["num_warmup_trials"]
    result_filename    = input_dict.get("results_filename")
    ts_mode            = input_dict["ts_mode"]
    log_filename       = input_dict.get("log_filename")

    # 4) Set up logger & sampler
    logger = get_logger(__name__, filename=log_filename)
    ts     = ThompsonSampler(mode=ts_mode)
    ts.set_hide_progress(hide_progress)
    ts.set_evaluator(evaluator)

    # 5) Load *all* synthons, filter to just this reaction, then group into slots
    from collections import defaultdict
    all_regs = []
    for fn in reagent_file if isinstance(reagent_file, (list,tuple)) else [reagent_file]:
        all_regs.extend(create_reagents(fn))

    # keep only those with matching reaction_id
    regs = [r for r in all_regs if r.reaction_id == reaction_id]
    if not regs:
        raise RuntimeError(f"No reagents found for reaction_id={reaction_id}")

    # group by synton_idx into slot-lists
    groups = defaultdict(list)
    for r in regs:
        groups[r.synton_idx].append(r)
    reagent_lists = [groups[i] for i in sorted(groups)]

    # hand the grouped slots to the sampler
    ts.reagent_lists   = reagent_lists
    ts.num_prods       = math.prod(len(x) for x in reagent_lists)
    ts._disallow_tracker = DisallowTracker([len(x) for x in reagent_lists])

    # rebuild the caches exactly as in read_reagents()
    ts.all_reagents = [r for slot in reagent_lists for r in slot]
    ts.reaction_map = build_reaction_map(ts.all_reagents)
    ts.n_sites      = len(reagent_lists)
    ts._means       = np.zeros(len(ts.all_reagents), dtype=float)
    ts._stds        = np.zeros(len(ts.all_reagents), dtype=float)
    ts._refresh_global_stats()

    ts.set_reaction(reaction_smarts)

    # 6) Warm-up & search
    ts.warm_up(num_warmup_trials=num_warmup_trials)
    out_list = ts.search(num_cycles=num_ts_iterations)

    # 7) Logging & save
    total_evals      = evaluator.counter
    percent_searched = total_evals / ts.get_num_prods() * 100
    logger.info(f"{total_evals} evaluations | {percent_searched:.3f}% of total")

    # write results
    out_df = pd.DataFrame(out_list, columns=["score", "SMILES", "Name"])
    if result_filename:
        out_df.to_csv(result_filename, index=False)
        logger.info(f"Saved results to: {result_filename}")

    # print top hits
    if not hide_progress:
        ascending = (ts_mode != "maximize")
        top10 = (
            out_df
            .sort_values("score", ascending=ascending)
            .drop_duplicates(subset="SMILES")
            .head(10)
        )
        print(top10)


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
