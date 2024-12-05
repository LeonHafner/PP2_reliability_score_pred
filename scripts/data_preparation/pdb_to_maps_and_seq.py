#!/usr/bin/env python3

"""
This script processes PDB files to extract protein sequences and/or
distance maps from specified chains. 

Given a directory of PDB files (with filenames indicating protein ID and chain ID), 
the script reads each file, identifies valid residues, and computes either:
- Protein sequences (as single-letter amino acid codes)
- Cα-Cα distance maps (as NumPy arrays)
- Both sequence and distance maps

The user can specify input and output paths, as well as choose whether to extract 
only sequences, only distance maps, or both. 

Command-line Arguments
----------------------
--input_dir : str (required)
    Path to the directory containing PDB files. Filenames are expected to follow 
    the format "<protein_id>-<chain_id>.pdb".
--output_distance_map : str (required)
    Path to the directory for saving computed distance map files (NumPy .npy format).
--output_sequence : str (required)
    Path to the CSV file where extracted protein sequences will be saved. Each line 
    of the CSV will contain "protein_id-chain_id,sequence".
--extract : {'sequence', 'distance_map', 'both'} (optional; default: 'both')
    Determines what data to extract from each PDB file:
    - 'sequence': Only extract the protein sequence.
    - 'distance_map': Only extract the distance map.
    - 'both': Extract both the sequence and the distance map.

Example Usage
-------------
    python3 pdb_to_maps_and_seq.py \
        --input_dir ./pdbs \
        --output_distance_map ./distmaps \
        --output_sequence ./sequences.csv \
        --extract both

Output
------
Depending on the chosen mode:
- If 'sequence' or 'both': A CSV file is generated with lines formatted as:
    protein_id-chain_id,SEQUENCE
- If 'distance_map' or 'both': For each PDB file processed, a corresponding 
    .npy file is created in the specified output directory, containing the 
    distance map as a 2D NumPy array.


The script raises ValueError if:
- Chains are not found.
- No valid residues are found.
- Sequence and distance map lengths do not match when both are requested.

Author: Leon Hafner
"""

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from Bio.PDB import PDBParser
from Bio.Data.PDBData import protein_letters_3to1

def get_valid_residues(chain):
    valid_residues = []
    for residue in chain:
        hetfield, resseq, icode = residue.id
        if hetfield == ' ' and residue.resname in protein_letters_3to1 and "CA" in residue:
            valid_residues.append(residue)
    return valid_residues


def sequence_from_residues(residues):
    sequence = []
    for residue in residues:
            sequence.append(protein_letters_3to1[residue.resname])
    return "".join(sequence)


def distance_map_from_residues(residues):
    ca_atoms = [residue["CA"].get_coord() for residue in residues]
    num_atoms = len(ca_atoms)
    distance_map = np.zeros((num_atoms, num_atoms))

    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            distance = np.linalg.norm(ca_atoms[i] - ca_atoms[j])
            distance_map[i, j] = distance_map[j, i] = distance

    return distance_map


parser = argparse.ArgumentParser(description="Process PDB files to generate distance maps and sequences.")
parser.add_argument(
    "--input_dir",
    type=str,
    help="Path to the directory containing PDB files.",
    required=True
    )
parser.add_argument(
    "--output_distance_map",
    type=str,
    help="Path to the directory for saving distance maps."
)
parser.add_argument(
    "--output_sequence",
    type=str,
    help="Path to the CSV file for saving sequences."
)
parser.add_argument(
    "--extract",
    type=str,
    choices=["sequence", "distance_map", "both"],
    default="both",
    help="Flag to specify what to extract: only 'sequence', only 'distance_map', or 'both'."
)

args = parser.parse_args()

# Check if required outputs are provided based on the mode
if args.extract in ["sequence", "both"] and not args.output_sequence:
    parser.error("--output_sequence is required when --extract is 'sequence' or 'both'.")

if args.extract in ["distance_map", "both"] and not args.output_distance_map:
    parser.error("--output_distance_map is required when --extract is 'distance_map' or 'both'.")


input_dir = args.input_dir
output_distance_map = args.output_distance_map
output_sequence = args.output_sequence
extract_mode = args.extract

if extract_mode in ["distance_map", "both"]:
    os.makedirs(output_distance_map, exist_ok=True)

sequences = []

for file_name in tqdm(os.listdir(input_dir)):
    if file_name.endswith('.pdb'):
        full_path = os.path.join(input_dir, file_name)

        if os.path.isfile(full_path):
            protein_id = os.path.basename(full_path).split('.')[0]
            if len(protein_id.split('-')) == 2:
                protein_id, chain_id = protein_id.split('-')
                parser = PDBParser(QUIET=True)
                structure = parser.get_structure("protein", full_path)

                chain = None
                for model in structure:
                    if chain_id in model:
                        chain = model[chain_id]
                        break

                if chain is None:
                    raise ValueError(f"Chain {chain_id} not found in the structure of {protein_id}-{chain_id}.")

                valid_residues = get_valid_residues(chain)

                if not valid_residues:
                    raise ValueError(f"No valid residues found in {protein_id}-{chain_id}.")

                sequence = sequence_from_residues(valid_residues) if extract_mode in ["sequence", "both"] else None
                distance_map = distance_map_from_residues(valid_residues) if extract_mode in ["distance_map", "both"] else None

                if extract_mode == "both":
                    if not (len(sequence) == distance_map.shape[0] == distance_map.shape[1]):
                        raise ValueError(f"Error processing {protein_id}-{chain_id}. Different lengths: {len(sequence)} vs. {distance_map.shape[0]}")

                if sequence is not None:
                    sequences.append((f"{protein_id}-{chain_id}", sequence))
                if distance_map is not None:
                    np.save(os.path.join(output_distance_map, f"{protein_id}-{chain_id}.npy"), distance_map)

if extract_mode in ["sequence", "both"]:
    with open(output_sequence, "w") as f:
        for pid, seq in sequences:
            f.write(f"{pid},{seq}\n")

print("Conversion finished!")
