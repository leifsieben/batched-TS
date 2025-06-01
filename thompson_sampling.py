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
from ts_utils import read_reagents
from ts_utils import create_reagents, build_reaction_map


class ThompsonSampler:
    def __init__(self, mode="maximize", log_filename: Optional[str] = None, output_csv: Optional[str] = None):
        """
        Basic init
        :param mode: maximize or minimize
        :param log_filename: Optional filename to write logging to. If None, logging will be output to stdout
        """
        self.reaction = None
        self.evaluator = None
        self.num_prods = 0
        self.logger = get_logger(__name__, filename=log_filename)
        self.logger.info(f"ThompsonSampler initialized with mode={mode}")
        self._disallow_tracker = None
        self.hide_progress = False
        self.output_csv = output_csv
        self._mode = mode
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

    def set_reactions(self, reaction_smarts_map: dict[str, str]) -> None:
        """
        Store one RDKit Reaction object per reaction_id.  reaction_smarts_map maps each
        reaction_id (matching those in self.rxn_ids) → SMARTS string.  We parse each SMARTS
        into an RDKit Reaction and save it in self._rxn_by_id.
        """
        from rdkit.Chem import AllChem

        self._rxn_by_id: dict[str, AllChem.ChemicalReaction] = {}
        for rxn_id, smarts in reaction_smarts_map.items():
            # Convert SMARTS to an RDKit Reaction object
            self._rxn_by_id[rxn_id] = AllChem.ReactionFromSmarts(smarts)


    def read_reagents(self, reagent_file_list: list[str], num_to_select: Optional[int] = None):
        """
        Reads one or more reagent files (tab-delimited, with columns SMILES, synton_id, synton#, reaction_id, release),
        groups all Reagent objects by reaction_id, and builds one DisallowTracker per reaction.

        After this call, the sampler will know:
          - self._reagents_by_rxn: dict[str, List[List[Reagent]]]
            mapping reaction_id → list of “slot-lists,” where each slot-list is a List[Reagent].
          - self._trackers_by_rxn: dict[str, DisallowTracker]
            mapping reaction_id → a DisallowTracker that tracks disallowed combinations for that reaction.
          - self.num_prods: total possible products across all reactions (sum over rxn of ∏ slot_sizes).
        """
        # 1) Load all Reagent objects from every file:
        all_regs: list[Reagent] = []
        for fname in reagent_file_list:
            regs = create_reagents(fname, num_to_select)
            all_regs.extend(regs)

        # 2) Build a reaction_map: reaction_id → (slot_index → List[Reagent])
        rxn_map: dict[str, dict[int, list[Reagent]]] = build_reaction_map(all_regs)

        # 3) Convert rxn_map into:
        #      self._reagents_by_rxn[rxn_id] = List[List[Reagent]] (slot 0, slot 1, …) sorted by slot index
        #    and build a DisallowTracker for each reaction_id.
        self._reagents_by_rxn: dict[str, list[list[Reagent]]] = {}
        self._trackers_by_rxn: dict[str, DisallowTracker] = {}
        self.rxn_ids: list[str] = []  # keep a list of reaction_ids for convenience

        total_products = 0
        for rxn_id, slot_dict in rxn_map.items():
            # Determine how many slots this reaction has:
            max_slot = max(slot_dict.keys())
            per_slot_lists: list[list[Reagent]] = []
            for slot_i in range(max_slot + 1):
                # If a slot index is missing, treat it as empty list (should not normally happen if data is well-formed)
                per_slot_lists.append(slot_dict.get(slot_i, []))

            # Save the reagent lists for this reaction:
            self._reagents_by_rxn[rxn_id] = per_slot_lists
            self.rxn_ids.append(rxn_id)

            # Build a DisallowTracker for exactly these slot sizes:
            slot_counts = [len(lst) for lst in per_slot_lists]
            self._trackers_by_rxn[rxn_id] = DisallowTracker(slot_counts)

            # Count how many products this reaction can enumerate:
            prod_count = math.prod(slot_counts) if slot_counts else 0
            total_products += prod_count

        self.num_prods = total_products
        self.logger.info(
            f"{self.num_prods:.2e} possible products across {len(self._reagents_by_rxn)} reactions"
        )

    def evaluate(self, rxn_id: str, choice_list: List[int]) -> Tuple[str, str, float]:
        """
        Evaluate a single reaction specified by rxn_id, using the chosen indices in choice_list.
        :param rxn_id: the reaction identifier (one of self.rxn_ids)
        :param choice_list: list of reagent-indices, one per slot in that reaction
        :return: (product_smiles, product_name, score)
        """
        # 1) Fetch the per-slot reagent-lists for this reaction:
        per_slot_lists = self._reagents_by_rxn[rxn_id]

        # 2) Build selected_reagents = [Reagent, Reagent, …] for each slot:
        selected_reagents: List[Reagent] = []
        for idx, reagent_idx in enumerate(choice_list):
            if reagent_idx < 0 or reagent_idx >= len(per_slot_lists[idx]):
                raise IndexError(f"Choice index {reagent_idx} out of range for slot {idx} in reaction {rxn_id}")
            selected_reagents.append(per_slot_lists[idx][reagent_idx])

        # 3) Run the RDKit reaction using the reaction object for this rxn_id
        rxn = self._rxn_by_id[rxn_id]
        prod = rxn.RunReactants([r.mol for r in selected_reagents])
        product_name = "_".join([r.reagent_name for r in selected_reagents])
        res = np.nan
        product_smiles = "FAIL"
        if prod:
            prod_mol = prod[0][0]
            Chem.SanitizeMol(prod_mol)
            product_smiles = Chem.MolToSmiles(prod_mol)
            res = self.evaluator.evaluate(prod_mol)
            if np.isfinite(res):
                # Record the score back on each reagent
                for r in selected_reagents:
                    r.add_score(res)

        return product_smiles, product_name, res

    def warm_up(self, num_warmup_trials: int = 3, batch_size: int = 128):
        """
        Warm-up all reactions:
        1) For each reaction_id, for each slot i, for each reagent j in that slot:
            • pick up to num_warmup_trials random partner combinations (skipping those
            disallowed by DisallowTracker)
            • immediately call tracker.update(...) on each full combination
            • buffer (reaction_id, full_choice_list) in all_candidates

        2) Batch-evaluate all buffered combinations in chunks of size batch_size, writing
            each (score, SMILES, name) to self.output_csv

        3) Compute global prior mean/std from all finite scores

        4) For each reagent that produced ≥1 finite score, call
            reagent.init_given_prior(prior_mean, prior_std).
        """
        self.logger.info(f"Starting warm-up: {num_warmup_trials} trials/reagent, batch_size={batch_size}")

        # Keep track of reagents that will need initialization
        reagents_to_initialize: dict[tuple[str, int, int], list[float]] = {}

        # 1) Build warm-up buffer and immediately call tracker.update(...) on every full selection
        all_candidates: list[tuple[str, list[int]]] = []
        
        for rxn_id in self.rxn_ids:
            per_slot_lists = self._reagents_by_rxn[rxn_id]
            tracker = self._trackers_by_rxn[rxn_id]
            n_slots = len(per_slot_lists)
            counts = [len(slot) for slot in per_slot_lists]
            
            # For each slot in this reaction
            for focal_slot in range(n_slots):
                partner_slots = [x for x in range(n_slots) if x != focal_slot]
                
                # For each reagent in the focal slot
                for focal_reagent_idx in range(counts[focal_slot]):
                    
                    # Generate num_warmup_trials combinations for this reagent
                    for trial in range(num_warmup_trials):
                        # Start with empty selection
                        local_sel = [DisallowTracker.Empty] * n_slots
                        
                        # Check if this focal reagent is allowed
                        local_sel[focal_slot] = DisallowTracker.To_Fill
                        disallow_mask = tracker.get_disallowed_selection_mask(local_sel)
                        if focal_reagent_idx in disallow_mask:
                            # This reagent can't be used, skip
                            continue
                        
                        # Fix the focal reagent
                        local_sel[focal_slot] = focal_reagent_idx
                        
                        # Fill partner slots randomly (respecting disallow constraints)
                        valid_combination = True
                        for partner_slot in partner_slots:
                            # Set this slot as To_Fill
                            local_sel[partner_slot] = DisallowTracker.To_Fill
                            partner_disallow = tracker.get_disallowed_selection_mask(local_sel)
                            
                            # Generate random selection scores
                            selection_scores = np.random.uniform(size=counts[partner_slot])
                            selection_scores[list(partner_disallow)] = np.nan
                            
                            # Check if any valid options remain
                            if np.all(np.isnan(selection_scores)):
                                valid_combination = False
                                break
                            
                            # Select random partner
                            chosen_partner = int(np.nanargmax(selection_scores))
                            local_sel[partner_slot] = chosen_partner
                        
                        if valid_combination:
                            # Update disallow tracker with this combination
                            tracker.update(local_sel)
                            # Store for batch evaluation
                            all_candidates.append((rxn_id, local_sel.copy()))

        if not all_candidates:
            raise RuntimeError("No warm-up candidates were generated.")
        
        self.logger.info(f"Generated {len(all_candidates)} warm-up candidates.")

        # 2) Open CSV for output
        import os, csv
        if self.output_csv is None:
            raise RuntimeError("No output_csv specified in ThompsonSampler; cannot write warm-up results.")
        
        write_header = not os.path.exists(self.output_csv)
        csv_file = open(self.output_csv, "a", newline="")
        csv_writer = csv.writer(csv_file)
        if write_header:
            csv_writer.writerow(["batch", "score", "smiles", "name"])


        # 3) Batch-evaluate all combinations
        warmup_results: list[tuple[str, str, float]] = []
        n_total = len(all_candidates)
        
        for start_idx in range(0, n_total, batch_size):
            end_idx = min(start_idx + batch_size, n_total)
            batch_candidates = all_candidates[start_idx:end_idx]
            
            # Build molecules for this batch
            batch_molecules = []
            batch_names = []
            batch_metadata = []  # Store (rxn_id, choice, reagent_objects) for each valid molecule
            
            for rxn_id, choice_list in batch_candidates:
                per_slot = self._reagents_by_rxn[rxn_id]
                selected_reagents = [per_slot[slot_i][choice_list[slot_i]] for slot_i in range(len(choice_list))]
                
                # Try to run the reaction
                rxn = self._rxn_by_id[rxn_id]
                products = rxn.RunReactants([r.mol for r in selected_reagents])
                
                if products:
                    try:
                        product_mol = products[0][0]
                        Chem.SanitizeMol(product_mol)
                        product_name = "_".join(r.reagent_name for r in selected_reagents)
                        
                        batch_molecules.append(product_mol)
                        batch_names.append(product_name)
                        batch_metadata.append((rxn_id, choice_list, selected_reagents))
                    except:
                        # Reaction failed, write failure immediately
                        warmup_results.append([np.nan, "FAIL", ""])
                        csv_writer.writerow([0, np.nan, "FAIL", ""])
                else:
                    # No products, write failure immediately
                    warmup_results.append([np.nan, "FAIL", ""])  
                    csv_writer.writerow([0, np.nan, "FAIL", ""])
            
            # Evaluate valid molecules in batch
            if batch_molecules:
                if hasattr(self.evaluator, 'evaluate_batch'):
                    scores = self.evaluator.evaluate_batch(batch_molecules)
                else:
                    # Fallback to individual evaluation
                    scores = [self.evaluator.evaluate(mol) for mol in batch_molecules]
                
                # Process results
                for mol, name, score, (rxn_id, choice_list, selected_reagents) in zip(
                    batch_molecules, batch_names, scores, batch_metadata
                ):
                    smiles = Chem.MolToSmiles(mol)
                    score_float = float(score)
                    
                    # Store result
                    warmup_results.append([score_float, smiles, name])
                    csv_writer.writerow([0, score_float, smiles, name])
                    
                    # Track reagents that produced finite scores for later initialization
                    if np.isfinite(score_float):
                        for slot_i, reagent in enumerate(selected_reagents):
                            key = (rxn_id, slot_i, choice_list[slot_i])
                            if key not in reagents_to_initialize:
                                reagents_to_initialize[key] = []
                            reagents_to_initialize[key].append(score_float)

        csv_file.close()

        # 4) Compute global prior statistics
        valid_scores = [score for score, _, _ in warmup_results if np.isfinite(score)]
        if not valid_scores:
            raise RuntimeError("No valid scores returned from warm-up.")
        
        prior_mean = float(np.mean(valid_scores))
        prior_std = float(np.std(valid_scores))
        self._warmup_std = prior_std
        
        self.logger.info(
            f"warmup score stats: cnt={len(valid_scores)}, "
            f"mean={prior_mean:.4f}, std={prior_std:.4f}, "
            f"min={np.min(valid_scores):.4f}, max={np.max(valid_scores):.4f}"
        )

        # 5) Initialize reagents that produced finite scores
        # Add all collected scores to each reagent first
        for (rxn_id, slot_i, reagent_idx), scores in reagents_to_initialize.items():
            reagent = self._reagents_by_rxn[rxn_id][slot_i][reagent_idx]
            for score in scores:
                reagent.add_score(score)
        
        # Then initialize with prior
        initialized_count = 0
        for (rxn_id, slot_i, reagent_idx) in reagents_to_initialize.keys():
            reagent = self._reagents_by_rxn[rxn_id][slot_i][reagent_idx]
            try:
                reagent.init_given_prior(prior_mean=prior_mean, prior_std=prior_std)
                initialized_count += 1
            except ValueError as e:
                self.logger.warning(f"Failed to initialize reagent {reagent.reagent_name}: {e}")

        self.logger.info(f"Warm-up completed. Initialized {initialized_count} reagents.")
        return warmup_results


    def search(
        self,
        ts_num_iterations: int = 25,
        batch_size: int = 128
    ) -> List[List]:
        """
        Run Thompson Sampling for ts_num_iterations rounds, each round generating batch_size new molecules,
        immediately writing each batch's results to CSV (columns: score, smiles, name), and returning a list
        of all [score, smiles, name].
        """
        out_list: List[List] = []
        rng = np.random.default_rng()

        # Open CSV and write header if not exists
        import os, csv
        write_header = not os.path.exists(self.output_csv)
        csv_file = open(self.output_csv, "a", newline="")
        csv_writer = csv.writer(csv_file)
        if write_header:
            csv_writer.writerow(["batch", "score", "smiles", "name"])

        for cycle in tqdm(range(ts_num_iterations), desc="Thompson rounds", disable=self.hide_progress):
            batch_molecules = []
            batch_names = []
            batch_metadata = []  # Store (rxn_id, choice_list, selected_reagents) for tracking

            # 1) Generate batch_size new molecules using Thompson Sampling with retry logic
            molecules_in_batch = 0
            attempts = 0
            max_total_attempts = batch_size * 5  # Reasonable limit to avoid infinite loops

            while molecules_in_batch < batch_size and attempts < max_total_attempts:
                attempts += 1
                
                # Step 1a: Choose reaction based on Thompson Sampling across all initialized reagents
                reaction_candidates = []
                
                for rxn_id in self.rxn_ids:
                    per_slot_lists = self._reagents_by_rxn[rxn_id]
                    
                    # Sample from all reagents in this reaction to get a "reaction score"
                    reaction_samples = []
                    for slot_reagents in per_slot_lists:
                        slot_samples = []
                        for reagent in slot_reagents:
                            if hasattr(reagent, 'current_mean') and hasattr(reagent, 'current_std'):
                                sample = rng.normal(reagent.current_mean, reagent.current_std)
                                slot_samples.append(sample)
                        
                        if slot_samples:
                            # Take best sample from this slot as representative
                            best_slot_sample = max(slot_samples) if self._mode.startswith("maximize") else min(slot_samples)
                            reaction_samples.append(best_slot_sample)
                    
                    if reaction_samples:
                        # Reaction score is sum/mean of best slot samples
                        reaction_score = np.mean(reaction_samples)
                        reaction_candidates.append((reaction_score, rxn_id))
                
                if not reaction_candidates:
                    # Fallback: random reaction selection if no reagents initialized
                    chosen_rxn = random.choice(self.rxn_ids)
                else:
                    # Choose reaction based on its aggregated Thompson sample
                    if self._mode.startswith("maximize"):
                        _, chosen_rxn = max(reaction_candidates, key=lambda x: x[0])
                    else:
                        _, chosen_rxn = min(reaction_candidates, key=lambda x: x[0])

                # Step 1b: Within chosen reaction, select reagents using Thompson Sampling
                per_slot_lists = self._reagents_by_rxn[chosen_rxn]
                tracker = self._trackers_by_rxn[chosen_rxn]
                n_slots = len(per_slot_lists)
                
                # Initialize selection array
                local_sel = [DisallowTracker.Empty] * n_slots
                
                # Fill slots in random order (preserving original TS logic)
                fill_order = list(range(n_slots))
                random.shuffle(fill_order)
                
                valid_combination = True
                for slot_i in fill_order:
                    # Set current slot to To_Fill
                    local_sel[slot_i] = DisallowTracker.To_Fill
                    disallow_mask = tracker.get_disallowed_selection_mask(local_sel)
                    
                    slot_reagents = per_slot_lists[slot_i]
                    
                    # Get current beliefs for reagents in this slot
                    mus = np.array([
                        r.current_mean if hasattr(r, 'current_mean') else 0.0 
                        for r in slot_reagents
                    ])
                    sigs = np.array([
                        r.current_std if hasattr(r, 'current_std') else 1.0 
                        for r in slot_reagents
                    ])
                    
                    # Thompson sampling: sample from beliefs
                    choice_samples = rng.normal(size=len(slot_reagents)) * sigs + mus
                    
                    # Mask out disallowed reagents
                    if disallow_mask:
                        choice_samples[np.array(list(disallow_mask))] = np.nan
                    
                    # Check if any valid options remain
                    if np.all(np.isnan(choice_samples)):
                        valid_combination = False
                        break
                    
                    # Select reagent using pick_function (argmax/argmin or Boltzmann)
                    chosen_idx = int(self.pick_function(choice_samples))
                    local_sel[slot_i] = chosen_idx
                
                if not valid_combination:
                    # Skip this attempt and try again
                    continue
                    
                # Step 1c: Update DisallowTracker and generate molecule
                tracker.update(local_sel)
                
                # Generate the molecule
                selected_reagents = [per_slot_lists[slot_i][local_sel[slot_i]] for slot_i in range(n_slots)]
                rxn = self._rxn_by_id[chosen_rxn]
                products = rxn.RunReactants([r.mol for r in selected_reagents])
                
                if products:
                    try:
                        product_mol = products[0][0]
                        Chem.SanitizeMol(product_mol)
                        product_name = "_".join(r.reagent_name for r in selected_reagents)
                        
                        batch_molecules.append(product_mol)
                        batch_names.append(product_name)
                        batch_metadata.append((chosen_rxn, local_sel.copy(), selected_reagents))
                        molecules_in_batch += 1  # Successfully generated a molecule
                    except:
                        # Reaction failed - still count as an attempt but don't add to batch
                        out_list.append([np.nan, "FAIL", ""])
                        csv_writer.writerow([cycle + 1, np.nan, "FAIL", ""])
                else:
                    # No products - still count as an attempt but don't add to batch
                    out_list.append([np.nan, "FAIL", ""])
                    csv_writer.writerow([cycle + 1, np.nan, "FAIL", ""])

            # Log if we couldn't fill the entire batch
            if molecules_in_batch < batch_size:
                self.logger.warning(f"Batch {cycle+1}: Only generated {molecules_in_batch}/{batch_size} molecules after {attempts} attempts")

            # 2) Batch-evaluate generated molecules
            if batch_molecules:
                if hasattr(self.evaluator, 'evaluate_batch'):
                    scores = self.evaluator.evaluate_batch(batch_molecules)
                else:
                    scores = [self.evaluator.evaluate(mol) for mol in batch_molecules]
                
                # 3) Process results and update reagent beliefs
                for mol, name, score, (rxn_id, choice_list, selected_reagents) in zip(
                    batch_molecules, batch_names, scores, batch_metadata
                ):
                    smiles = Chem.MolToSmiles(mol)
                    score_float = float(score)
                    
                    # Store result
                    out_list.append([score_float, smiles, name])
                    csv_writer.writerow([cycle + 1, score_float, smiles, name])
                    
                    # CRITICAL: Update reagent beliefs if score is finite
                    if np.isfinite(score_float):
                        for reagent in selected_reagents:
                            reagent.add_score(score_float)

            # 4) Log progress every 100 batches
            if (cycle + 1) % 100 == 0 and out_list:
                finite_results = [x for x in out_list if np.isfinite(x[0])]
                if finite_results:
                    top_score, top_smi, top_name = self._top_func(finite_results, key=lambda x: x[0])
                    self.logger.info(
                        f"After {cycle+1} batches ({(cycle+1)*batch_size:,} samples), "
                        f"top_score={top_score:.3f}, smiles={top_smi}, name={top_name}"
                    )

        csv_file.close()
        return out_list