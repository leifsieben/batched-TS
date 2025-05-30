from typing import List, Optional
from collections import defaultdict
from reagent import Reagent

def create_reagents(filename: str,
                    num_to_select: Optional[int] = None
                   ) -> List[Reagent]:
    """
    Reads a tab-delimited file with columns:
       SMILES, synton_id (reagent name), synton# (slot index),
       reaction_id, release
    and returns a list of Reagent(...) objects.
    """
    reagent_list: List[Reagent] = []
    with open(filename, 'r') as f:
        raw_header = f.readline().rstrip('\n').split('\t')
        # find the columns
        idx_smiles     = raw_header.index("SMILES")
        idx_reagent_nm = raw_header.index("synton_id")
        idx_synton     = raw_header.index("synton#")
        idx_reaction   = raw_header.index("reaction_id")
        # (we ignore "release" here)

        for line in f:
            cols = line.rstrip('\n').split('\t')
            smi       = cols[idx_smiles]
            name      = cols[idx_reagent_nm]
            syn_idx   = int(cols[idx_synton])
            rxn_id    = cols[idx_reaction]

            reagent_list.append(
                Reagent(
                    reagent_name = name,
                    smiles       = smi,
                    reaction_id  = rxn_id,
                    synton_idx   = syn_idx,
                )
            )
            if num_to_select and len(reagent_list) >= num_to_select:
                break

    return reagent_list


def read_reagents(single_file: str,
                  num_to_select: Optional[int] = None
                 ) -> List[List[Reagent]]:
    """
    Read *one* tab-delimited file with a synton_idx column, and
    return a list of reagent‐lists, one per synton_idx.
    """
    all_regs = create_reagents(single_file, num_to_select)

    # group by synton_idx
    groups: dict[int, List[Reagent]] = defaultdict(list)
    for r in all_regs:
        groups[r.synton_idx].append(r)

    # ensure deterministic slot order 0,1,2,...
    reagent_lists = [groups[i] for i in sorted(groups)]
    return reagent_lists




def build_reaction_map(all_reagents: list[Reagent]) -> dict[str, dict[int, list[Reagent]]]:
    """
    Builds a fast lookup table to find all Reagents for a given reaction ID. 
    :param all_reagents: a list of Reagents.
    :return: Map of reaction ID to Reagents. 
    """
    reaction_map: dict[str, dict[int, list[Reagent]]] = defaultdict(lambda: defaultdict(list))
    for r in all_reagents:
        reaction_map[r.reaction_id][r.synton_idx].append(r)
    return reaction_map
