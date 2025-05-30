import random
from typing import List, Optional, Tuple

import functools
import math
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm.auto import tqdm

from disallow_tracker import DisallowTracker
from reagent import Reagent
from ts_logger import get_logger
from ts_utils import build_reaction_map, create_reagents, read_reagents



class ThompsonSampler:
    def __init__(self, mode="maximize", log_filename: Optional[str] = None):
        """
        Basic init
        :param mode: maximize or minimize
        :param log_filename: Optional filename to write logging to. If None, logging will be output to stdout
        """
        # A list of lists of Reagents. Each component in the reaction will have one list of Reagents in this list
        print("Initialized ThompsonSampler", flush=True)
        self.reagent_lists: List[List[Reagent]] = []
        self.reaction = None
        self.evaluator = None
        self.num_prods = 0
        self.logger = get_logger(__name__, filename=log_filename)
        self._disallow_tracker = None
        self.hide_progress = False
        self._mode = mode
        self.rng = np.random.default_rng()
        if self._mode == "maximize":
            self.pick_function = np.nanargmax
            self._top_func = max
        elif self._mode == "minimize":
            self.pick_function = np.nanargmin
            self._top_func = min
        elif self._mode == "maximize_boltzmann":
            # See documentation for _boltzmann_reweighted_pick
            self.pick_function = functools.partial(self._boltzmann_reweighted_pick)
            self._top_func = max
        elif self._mode == "minimize_boltzmann":
            # See documentation for _boltzmann_reweighted_pick
            self.pick_function = functools.partial(self._boltzmann_reweighted_pick)
            self._top_func = min
        else:
            raise ValueError(f"{mode} is not a supported argument")
        self._warmup_std = None

    def _refresh_global_stats(self):
        """Copy each reagent’s current_mean/std into our flat arrays."""
        for i, r in enumerate(self.all_reagents):
            self._means[i] = r.current_mean
            self._stds[i]  = r.current_std

    def _boltzmann_reweighted_pick(self, scores: np.ndarray):
        """Rather than choosing the top sampled score, use a reweighted probability.

        Zhao, H., Nittinger, E. & Tyrchan, C. Enhanced Thompson Sampling by Roulette
        Wheel Selection for Screening Ultra-Large Combinatorial Libraries.
        bioRxiv 2024.05.16.594622 (2024) doi:10.1101/2024.05.16.594622
        suggested several modifications to the Thompson Sampling procedure.
        This method implements one of those, namely a Boltzmann style probability distribution
        from the sampled values. The reagent is chosen based on that distribution rather than
        simply the max sample.
        """
        if self._mode == "minimize_boltzmann":
            scores = -scores
        exp_terms = np.exp(scores / self._warmup_std)
        probs = exp_terms / np.nansum(exp_terms)
        probs[np.isnan(probs)] = 0.0
        return np.random.choice(probs.shape[0], p=probs)

    def set_hide_progress(self, hide_progress: bool) -> None:
        """
        Hide the progress bars
        :param hide_progress: set to True to hide the progress baars
        """
        self.hide_progress = hide_progress

    def read_reagents(self, reagent_file_list, num_to_select: Optional[int] = None):
        """
        Reads the reagents from reagent_file_list
        :param reagent_file_list: List of reagent filepaths
        :param num_to_select: Max number of reagents to select from the reagents file (for dev purposes only)
        :return: None
        """
        self.reagent_lists = read_reagents(reagent_file_list, num_to_select)
        self.num_prods = math.prod([len(x) for x in self.reagent_lists])
        self.logger.info(f"{self.num_prods:.2e} possible products")
        self._disallow_tracker = DisallowTracker([len(x) for x in self.reagent_lists])
        # now that reagents are loaded, build flattened indices & caches
        self.all_reagents = [r for slot in self.reagent_lists for r in slot]
        self.reaction_map = build_reaction_map(self.all_reagents)
        self.n_sites = max(r.synton_idx for r in self.all_reagents) + 1

        # global flat caches
        self._means = np.zeros(len(self.all_reagents), dtype=float)
        self._stds  = np.zeros(len(self.all_reagents), dtype=float)
        self._refresh_global_stats()

        # per-slot caches
        self.slot_means = []
        self.slot_stds  = []
        for slot in self.reagent_lists:
            self.slot_means.append(np.zeros(len(slot), dtype=float))
            self.slot_stds.append (np.zeros(len(slot), dtype=float))
        self._refresh_slot_stats()

    def _refresh_slot_stats(self):
        """Copy each reagent’s current_mean/std into per-slot arrays."""
        for slot_idx, slot in enumerate(self.reagent_lists):
            means = self.slot_means[slot_idx]
            stds  = self.slot_stds[slot_idx]
            for i, r in enumerate(slot):
                means[i] = r.current_mean
                stds [i] = r.current_std

    def evaluate_batch(self, choices: List[List[int]]) -> List[Tuple[str,str,float]]:
        """
        Batch‐evaluate many reagent combinations in one go.
        Gaussian updates commute, so calling add_score per appearance
        yields the same posterior as a single aggregated update.
        """
        results = []
        # if evaluator supports batch, use it
        has_batch = hasattr(self.evaluator, "evaluate_batch")
        for choice_list in choices:
            # build product & name
            reagents = [self.reagent_lists[i][c] for i,c in enumerate(choice_list)]
            prod = self.reaction.RunReactants([r.mol for r in reagents])[0][0]
            Chem.SanitizeMol(prod)
            smiles = Chem.MolToSmiles(prod)
            name = "_".join(r.reagent_name for r in reagents)
            # score
            if has_batch:
                score = self.evaluator.evaluate_batch([prod])[0]
            else:
                score = self.evaluator.evaluate(prod)
            # update priors
            if np.isfinite(score):
                for r in reagents:
                    r.add_score(float(score))
            results.append((smiles, name, float(score)))
        return results


    def get_num_prods(self) -> int:
        """
        Get the total number of possible products
        :return: num_prods
        """
        return self.num_prods

    def set_evaluator(self, evaluator):
        """
        Define the evaluator
        :param evaluator: evaluator class, must define an evaluate method that takes an RDKit molecule
        """
        self.evaluator = evaluator

    def set_reaction(self, rxn_smarts):
        """
        Define the reaction
        :param rxn_smarts: reaction SMARTS
        """
        self.reaction = AllChem.ReactionFromSmarts(rxn_smarts)

    def evaluate(self, choice_list: List[int]) -> Tuple[str, str, float]:
        """Evaluate a set of reagents."""
        selected = [
            self.reagent_lists[i][choice_list[i]]
            for i in range(len(choice_list))
        ]
        prods = self.reaction.RunReactants([r.mol for r in selected])
        product_name = "_".join(r.reagent_name for r in selected)
        if not prods:
            return "FAIL", product_name, np.nan

        prod_mol = prods[0][0]
        Chem.SanitizeMol(prod_mol)
        product_smiles = Chem.MolToSmiles(prod_mol)

        score = float(self.evaluator.evaluate(prod_mol))
        if np.isfinite(score):
            for r in selected:
                r.add_score(score)

        return product_smiles, product_name, score

    def warm_up(self, num_warmup_trials=3):
        """Warm-up phase, each reagent is sampled with num_warmup_trials random partners
        :param num_warmup_trials: number of times to sample each reagent
        """
        print("Starting warmup.", flush=True)
        # get the list of reagent indices
        idx_list = list(range(0, len(self.reagent_lists)))
        # get the number of reagents for each component in the reaction
        reagent_count_list = [len(x) for x in self.reagent_lists]
        warmup_results = []
        for i in idx_list:
            partner_list = [x for x in idx_list if x != i]
            # The number of reagents for this component
            current_max = reagent_count_list[i]
            # For each reagent...
            for j in tqdm(range(0, current_max), desc=f"Warmup {i + 1} of {len(idx_list)}", disable=self.hide_progress):
                # For each warmup trial...
                for k in range(0, num_warmup_trials):
                    current_list = [DisallowTracker.Empty] * len(idx_list)
                    current_list[i] = DisallowTracker.To_Fill
                    disallow_mask = self._disallow_tracker.get_disallowed_selection_mask(current_list)
                    if j not in disallow_mask:
                        ## ok we can select this reagent
                        current_list[i] = j
                        # Randomly select reagents for each additional component of the reaction
                        for p in partner_list:
                            # tell the disallow tracker which site we are filling
                            current_list[p] = DisallowTracker.To_Fill
                            # get the new disallow mask
                            disallow_mask = self._disallow_tracker.get_disallowed_selection_mask(current_list)
                            selection_scores = np.random.uniform(size=reagent_count_list[p])
                            # null out the disallowed ones
                            selection_scores[list(disallow_mask)] = np.nan
                            # and select a random one
                            current_list[p] = np.nanargmax(selection_scores).item(0)
                        self._disallow_tracker.update(current_list)
                        product_smiles, product_name, score = self.evaluate(current_list)
                        if np.isfinite(score):
                            warmup_results.append([score, product_smiles, product_name])

        warmup_scores = [ws[0] for ws in warmup_results]
        self.logger.info(
            f"warmup score stats: "
            f"cnt={len(warmup_scores)}, "
            f"mean={np.mean(warmup_scores):0.4f}, "
            f"std={np.std(warmup_scores):0.4f}, "
            f"min={np.min(warmup_scores):0.4f}, "
            f"max={np.max(warmup_scores):0.4f}")
        # initialize each reagent
        prior_mean = np.mean(warmup_scores)
        prior_std = np.std(warmup_scores)
        self._warmup_std = prior_std
        for i in range(0, len(self.reagent_lists)):
            for j in range(0, len(self.reagent_lists[i])):
                reagent = self.reagent_lists[i][j]
                try:
                    reagent.init_given_prior(prior_mean=prior_mean, prior_std=prior_std)
                except ValueError:
                    self.logger.info(f"Skipping reagent {reagent.reagent_name} because there were no successful evaluations during warmup")
                    self._disallow_tracker.retire_one_synthon(i, j)
        self.logger.info(f"Top score found during warmup: {max(warmup_scores):.3f}")
        self._refresh_global_stats()
        print("End of warmup.", flush=True)
        return warmup_results

    def search(self, num_cycles=25, batch_size=1, log_every=100):
        """Run the search in batches of `batch_size`."""
        out = []
        buffer = []
        rng = self.rng

        for i in tqdm(range(num_cycles), desc="Cycle", disable=self.hide_progress):
            # --- propose one full combination (updating disallow_tracker) ---
            sel = [DisallowTracker.Empty] * len(self.reagent_lists)
            for slot_id in random.sample(range(len(self.reagent_lists)), len(self.reagent_lists)):
                sel[slot_id] = DisallowTracker.To_Fill
                dis_mask = self._disallow_tracker.get_disallowed_selection_mask(sel)

                # use cached slot stats here:
                mus  = self.slot_means[slot_id]
                sigs = self.slot_stds [slot_id]
                samples = rng.normal(mus, sigs)
                if dis_mask:
                    samples[np.array(list(dis_mask))] = np.nan

                sel[slot_id] = int(self.pick_function(samples))

            # retire / disallow this synthon choice
            self._disallow_tracker.update(sel)

            buffer.append(sel)

            # when buffer full (or last iteration), score the batch
            if len(buffer) == batch_size or i == num_cycles - 1:
                # evaluate_batch updates priors under the hood
                batch_results = self.evaluate_batch(buffer)
                # refresh our cached Gaussian stats
                self._refresh_global_stats()
                self._refresh_slot_stats()

                # collect only the finite‐score results
                for score, smiles, name in batch_results:
                    if np.isfinite(score):
                        out.append([score, smiles, name])

                buffer.clear()

            # optional logging
            if (i + 1) % log_every == 0 and out:
                top_score, top_smiles, top_name = max(out, key=lambda x: x[0])
                self.logger.info(f"Iteration {i+1}: top score={top_score:.3f} smiles={top_smiles}")
                print(f"Iteration {i+1}: top score={top_score:.3f} smiles={top_smiles}", flush=True)

        return out


# Additional methods for batched sampling

    def _propose_molecule(self) -> List[Reagent]:
        """
        Draws one full combination of synthons (first unconstrained,
        then reaction-constrained), updates disallow tracker, but
        does *not* call evaluator or update priors.
        """
        # Helper to sample and pick an index via current pick_function
        def _draw_pick(means, stds, mask=None):
            samples = self.rng.normal(means, stds)
            if mask is not None and mask.size:
                samples[mask] = np.nan
            idx = self.pick_function(samples)
            if np.isnan(idx):            # everything masked → no pick
                return None
            return int(idx)

        # 1) Unconstrained first pick over entire reagent pool (vectorized)
        samples = self.rng.normal(self._means, self._stds)
        idx0    = self.pick_function(samples)
        r0 = self.all_reagents[idx0]

        combo = {r0.synton_idx: r0}
        # mark this pick in tracker and retire if exhausted
        partial = [DisallowTracker.Empty] * self.n_sites
        partial[r0.synton_idx] = idx0
        self._disallow_tracker.update(partial)
        self._disallow_tracker.retire_one_synthon(r0.synton_idx, idx0)

        # 2) Fill remaining slots for this reaction
        for pos, slot_list in self.reaction_map[r0.reaction_id].items():
            if pos == r0.synton_idx:
                continue
            # build means and stds for this slot
            mus = np.array([r.current_mean for r in slot_list])
            sigs = np.array([r.current_std  for r in slot_list])
            # build partial mask for disallowed indices
            partial = [DisallowTracker.Empty] * self.n_sites
            for p, r_obj in combo.items():
                idx_in_slot = slot_list.index(r_obj)
                partial[p] = idx_in_slot
            partial[pos] = DisallowTracker.To_Fill
            mask = self._disallow_tracker.get_disallowed_selection_mask(partial)
            # pick using same logic, applying mask
            idx = _draw_pick(mus, sigs, mask)

            chosen = slot_list[idx]
            combo[pos] = chosen
            partial[pos] = idx
            self._disallow_tracker.update(partial)
            self._disallow_tracker.retire_one_synthon(pos, idx)

        # return reagents in slot order
        return [combo[i] for i in sorted(combo)]
    
    def enumerate_molecule(self) -> str:
        """
        Propose one molecule and return its SMILES; tracker is updated
        exactly as before.
        """
        reagents = self._propose_molecule()
        prod = self.reaction.RunReactants([r.mol for r in reagents])[0][0]
        smiles = Chem.MolToSmiles(prod)
        # note: retirement of fully-exhausted reagents is still handled by
        # self._disallow_tracker inside _propose_molecule()
        return smiles

    def sample_batch(self, batch_size: int) -> List[str]:
        batch, attempts = [], 0
        while len(batch) < batch_size and attempts < batch_size*10:
            mol = self._propose_molecule()
            attempts += 1
            if mol:
                smiles = Chem.MolToSmiles(self.reaction.RunReactants([r.mol for r in mol])[0][0])
                batch.append(smiles)
        return batch