""" Class for random sampling without replacement from a combinatorial library"""
import itertools
import random
from collections import defaultdict
from typing import DefaultDict, Set, Tuple

import numpy as np


class DisallowTracker:
    Empty = -1  # a sentinel value meaning fill this reagent spot - used in selection
    To_Fill = None  # a sentinel value meaning this reagent spot is open - used in selection

    def __init__(self, reagent_counts: list[int]):
        """
        Basic Init
        :param reagent_counts: A list of the number of reagents for each site of diversity in the reaction

        For example if the library to search has 3 reactants the first with 10 options, the second with 20
        and the third with 34 then reagent_counts would be the list [10, 20, 34]
        """

        self._initial_reagent_counts = np.array(reagent_counts)
        self._reagent_exhaust_counts = self._get_reagent_exhaust_counts()

        # this is where we keep track of the disallowed combinations
        self._disallow_mask: DefaultDict[Tuple[int | None], Set] = defaultdict(set)
        self._n_sampled = 0 ## number of products sampled
        self._total_product_size = np.prod(reagent_counts)

    @property
    def n_cycles(self) -> int:
        """ How many cycles are then in this reaction"""
        return len(self._initial_reagent_counts)

    def get_disallowed_selection_mask(self, current_selection: list[int | None]) -> Set[int]:
        """ Returns the disallowed reagents given the current_selection
        :param current_selection:  list of ints denoting the current selection
        :return: set[int] of the indices that are disallowed

        Current_selection is of length "n_cycles" for the current reaction and filled at each position
        with either Disallow_tracker.Empty, Disallow_tracker.To_Fill, or int >= 0
        additionally, Disallow_tracker.To_Fill can appear only once.
        """

        """Returns the disallowed reagents given the current"""

        if len(current_selection) != self.n_cycles:
            raise ValueError(f"current_selection must be equal in length to number of sites "
                             f"({self.n_cycles} for reaction")
        if len([v for v in current_selection if v == DisallowTracker.To_Fill]) != 1:
            raise ValueError(f"current_selection must have exactly one To_Fill slot.")

        return self._disallow_mask[tuple(current_selection)]

    def retire_one_synthon(self, cycle_id: int, synthon_index: int):
        print("Retiring synthon", cycle_id, synthon_index, flush=True)
        retire_mask = [self.Empty] * self.n_cycles
        retire_mask[cycle_id] = synthon_index
        self._retire_synthon_mask(retire_mask=retire_mask)

    def _retire_synthon_mask(self, retire_mask: list[int]):
        # If there are no Empty slots, we have a fully specified combination:
        if retire_mask.count(self.Empty) == 0:
            # We sampled one complete product → update counters & disallow table
            self._n_sampled += 1
            self._update(retire_mask)
        else:
            # For every position that is still Empty, create a new mask that marks it To_Fill
            for cycle_id in [i for i in range(self.n_cycles) if retire_mask[i] == self.Empty]:
                # Make a fresh copy so we don’t overwrite retire_mask in one branch
                next_mask = retire_mask.copy()
                next_mask[cycle_id] = self.To_Fill

                # Compute which synthons at this cycle are now disallowed
                ts_locations = np.ones(shape=self._initial_reagent_counts[cycle_id])
                disallowed_selections = self.get_disallowed_selection_mask(next_mask)
                if len(disallowed_selections):
                    ts_locations[np.array(list(disallowed_selections))] = np.nan

                # For each allowed synthon index, set it and recurse on a fresh copy
                for synthon_idx in np.argwhere(~np.isnan(ts_locations)).flatten():
                    deeper_mask = next_mask.copy()
                    deeper_mask[cycle_id] = synthon_idx
                    self._retire_synthon_mask(retire_mask=deeper_mask)


    def update(self, selected: list[int | None]) -> None:
        """
        Updates the disallow tracker with the selected reagents.
        :param selected: list[int]

        This means that this particular reagent combination will not be sampled again.

        Selected is the list of indexes that maps to what reagent was used at what position
        For example selected = [4, 5, 3]
              means reagent 4 at position 0
                    reagent 5 at position 1
                    reagent 3 at position 2
              will not be sampled again

        Two sentinel values are used in this routine:
            to_fill = None
            empty = -1
        """
        if len(selected) != self.n_cycles:
            msg = f"DisallowTracker selected size {len(selected)} but reaction has {self.n_cycles} sites of diversity"
            raise ValueError(msg)
        for site_id, sel, max_size in zip(list(range(self.n_cycles)), selected, self._initial_reagent_counts):
            if sel is not None and sel >= max_size:
                raise ValueError(f"Disallowed given index {sel} for site {site_id} which has {max_size} reagents")

        # all ok so call the internal update
        self._update(selected)

    def sample(self) -> list[int]:
        """ Randomly sample from the reaction without replacement"""
        if self._n_sampled == self._total_product_size:
            raise ValueError(f"Sampled {self._n_sampled} of {self._total_product_size} products in reaction - "
                             f"there are no more left to sample")
        selection_mask: list[int | None] = [self.Empty] * self.n_cycles
        selection_order: list[int] = list(range(self.n_cycles))
        random.shuffle(selection_order)
        for cycle_id in selection_order:
            selection_mask[cycle_id] = DisallowTracker.To_Fill
            selection_candidate_scores = np.random.uniform(size=self._initial_reagent_counts[cycle_id])
            selection_candidate_scores[list(self._disallow_mask[tuple(selection_mask)])] = np.NaN
            selection_mask[cycle_id] = np.nanargmax(selection_candidate_scores).item(0)
        self.update(selection_mask)
        self._n_sampled += 1
        return selection_mask

    def _get_reagent_exhaust_counts(self) -> dict[tuple[int,], int]:
        """
        Returns a dictionary denoting when reagents for a site are exhausted.

        For example is the reagent counts are [3,4,5] then the dictionary looks like this:
            {(0,): 20, (1,): 15, (2,): 12, (0, 1): 5, (0, 2): 4, (1, 2): 3}
        which means that a *particular* reagent at:
            site 0 is exhausted if it has been sampled in 20 (4*5) molecules,
            site 1 is exhausted if it has been sampled in 15 (3*5) molecules,
            ...
        also:
            A *particular pair* of reagents used in sites (0,2) are exhausted if they have been sampled 4 times
            and so on for the other reagents sites

        :return: dict[tuple[int,], int]
        """
        s = range(self.n_cycles)
        all_set = {*list(range(self.n_cycles))}
        power_set = itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(1, self.n_cycles))
        return {p: np.prod(self._initial_reagent_counts[list(all_set - {*list(p)})]) for p in power_set}

    def _update(self, selected: list[int | None]):
        """ Does the updates to the disallow masks w/o the error checking of parameters"""
        # ok now start the disallow fun
        for idx, value in enumerate(selected):
            # save what reagent_index was used at current position and set this
            # index to 'to_fill' then update the dictionary index by selected (w/None)
            # to include the value - meaning that this value can not be selected if the remaining
            # synthons are selected
            #   [4, 5, 3] -> [None, 5, 3]  then set indexed by disallow_mask[[None,5,3]] gets 'value' added
            #    meaning that 4 can not be selected as a reagent when [None, 5, 3] is selected in the select step
            selected[idx] = self.To_Fill
            if value is not None and value not in self._disallow_mask[tuple(selected)]:
                if value != self.Empty:
                    self._disallow_mask[tuple(selected)].add(value)
                    # now we get the counts to see if we need to retire a reagent so
                    # get the key, and then get the count for that key
                    count_key = tuple([r_pos for r_pos, r_idx in enumerate(selected) if r_idx != self.To_Fill])
                    if self._reagent_exhaust_counts[count_key] == len(self._disallow_mask[tuple(selected)]):
                        # Here comes the confusing part.  If we have exhausted a reagent then we need
                        # to update the disallow for 'empty' pairs.
                        # Let say we are updating [None, 5, 3] and have exhausted pair 5,3 then we need to
                        # say that if we get the pattern [empty, 5, to_fill] during the select step
                        # meaning that we spot 0 is empty, 1 has reagent id 5 and 2 is currently being selected
                        # that we do not select a 3 for position 2 or a 5 for position 1.
                        # The following recursive call deals with that.

                        self._update([self.Empty if v == self.To_Fill else v for v in selected])
            selected[idx] = value

    def is_reagent_exhausted(self, slot_idx: int, reagent_idx: int) -> bool:
        """
        Check if a specific reagent at a specific slot is exhausted.
        A reagent is exhausted if selecting it leads to no valid combinations.
        """
        # Create a selection pattern with this reagent fixed and all others Empty
        test_selection = [DisallowTracker.Empty] * self.n_cycles
        test_selection[slot_idx] = reagent_idx
        
        # Check all possible ways to fill the remaining slots
        remaining_slots = [i for i in range(self.n_cycles) if i != slot_idx]
        
        # If this is a single-slot reaction, check if this reagent has been used
        if not remaining_slots:
            single_pattern = tuple(test_selection)
            return reagent_idx in self._disallow_mask.get(single_pattern, set())
        
        # For multi-slot reactions, check if any valid combinations exist
        def can_fill_slots(selection_so_far: list, slots_to_fill: list) -> bool:
            if not slots_to_fill:
                # All slots filled - check if this combination is allowed
                return tuple(selection_so_far) not in self._disallow_mask or \
                    len(self._disallow_mask[tuple(selection_so_far)]) == 0
            
            current_slot = slots_to_fill[0]
            remaining_slots = slots_to_fill[1:]
            
            # Try each reagent at this slot
            for candidate_reagent in range(self._initial_reagent_counts[current_slot]):
                test_sel = selection_so_far.copy()
                test_sel[current_slot] = DisallowTracker.To_Fill
                
                # Check if this candidate is disallowed
                disallowed = self._disallow_mask.get(tuple(test_sel), set())
                if candidate_reagent not in disallowed:
                    # Try this reagent and recurse
                    test_sel[current_slot] = candidate_reagent
                    if can_fill_slots(test_sel, remaining_slots):
                        return True
            
            return False
        
        # Check if any valid combination exists with this reagent
        return not can_fill_slots(test_selection, remaining_slots)            