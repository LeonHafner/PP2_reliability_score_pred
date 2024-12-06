#!/usr/bin/env python3

"""
This script computes and processes contact maps from protein sequences using the ESM-2 model.
Given an input CSV file of protein IDs and sequences, it:
    1. Loads a pretrained ESM-2 model and its associated alphabet.
    2. Computes the Jacobian matrix of logits w.r.t. amino acid substitutions.
    3. Derives a contact map by measuring the change in logits and applying optional
       centering, APC correction, and symmetrization.
    4. Saves the resulting contact maps as .npy files in the specified output directory.

Args:
    --input_csv (str): Path to input CSV with protein IDs and sequences.
    --outdir (str): Path to output directory for the resulting contact map files.
    --apc (flag): Apply APC correction to the contact maps if specified.
    --sym (flag): Symmetrize the contact maps if specified.
    --center (flag): Center the Jacobian matrix before processing if specified.

Author: Leon Hafner
"""

import os
import argparse
from tqdm import tqdm
import torch
import numpy as np
import esm


def get_jacobian(seq, model, alphabet, device):
    model = model.to(device)
    standard_amino_acids = 'LAGVSERTIDPKQNFYMHWC'
    token_to_idx = {tok: i for i, tok in enumerate(alphabet.all_toks)}
    aa_indices = [token_to_idx[aa] for aa in standard_amino_acids]
    
    batch_converter = alphabet.get_batch_converter()
    data = [("stub_id", seq)]
    batch_tokens = batch_converter(data)[2].to(device)

    with torch.no_grad():
        token_logits = model(batch_tokens, repr_layers=[], return_contacts=False)["logits"]
    
    # Remove start and end tokens ([CLS] and [EOS]) and get standard aa_indices
    logits_original = token_logits[0, 1:-1, aa_indices]

    L, A = logits_original.shape

    J = torch.zeros(L, A, L, A, device=device)
    for i in range(L):
        substituted_sequences = []
        for s_aa in standard_amino_acids:
            seq_list = list(seq)
            seq_list[i] = s_aa
            substituted_sequences.append((f"protein_subst_{i}_{s_aa}", ''.join(seq_list)))

        batch_tokens_subst = batch_converter(substituted_sequences)[2].to(device)

        with torch.no_grad():
            results_subst = model(batch_tokens_subst, repr_layers=[], return_contacts=False)
            token_logits_subst = results_subst["logits"]
        
        # Remove start and end tokens ([CLS] and [EOS]) and get standard aa_indices
        logits_subst = token_logits_subst[:, 1:-1, aa_indices]

        for s_idx in range(A):
            delta_logits = logits_subst[s_idx] - logits_original
            J[i, s_idx, :, :] = delta_logits
    
    return J


def get_contact_map(jacobian, center=False, apc=False, sym=False):
    if center:
        for i in range(4):
            jacobian = jacobian - jacobian.mean(dim=i, keepdim=True)
    
    contact_map = torch.sqrt(torch.sum(jacobian ** 2, dim=(1, 3))).fill_diagonal_(0)

    if apc:
        contact_map = apply_apc(contact_map).fill_diagonal_(0)
    
    if sym:
        contact_map = (contact_map + contact_map.T) / 2
    
    return contact_map


def apply_apc(contact_map):
    mean_row = contact_map.mean(dim=0, keepdim=True)
    mean_col = contact_map.mean(dim=1, keepdim=True)
    mean_total = contact_map.mean()
    apc = (mean_row * mean_col) / mean_total
    return contact_map - apc


parser = argparse.ArgumentParser(description="Process protein contact maps.")

parser.add_argument("--input_csv", type=str, help="Path to the input CSV file containing protein sequences.", required=True)
parser.add_argument("--outdir", type=str, help="Path to the output directory where the maps are stored. Created if not existing", required=True)
parser.add_argument("--apc", action="store_true", help="Apply APC correction to contact maps.")
parser.add_argument("--sym", action="store_true", help="Symmetrize the contact map.")
parser.add_argument("--center", action="store_true", help="Center the Jacobian matrix before processing.")

args = parser.parse_args()

print(f"Start Processing: {args.input_csv}")
print(f"Output directory: {args.outdir}")
print(f"Centering: {args.center}")
print(f"APC: {args.apc}")
print(f"Symmetrizing: {args.sym}")

if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)

print("Loading model...")
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model = model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on {device}")

with open(args.input_csv, "r") as f:
    lines = [line.strip().split(',') for line in f.readlines() if line.strip()]

for pid, seq in tqdm(lines):
    jacobian = get_jacobian(seq, model, alphabet, device)
    
    contact_map = get_contact_map(jacobian, center=args.center, apc=args.apc, sym=args.sym)

    if isinstance(contact_map, torch.Tensor):
        contact_map = contact_map.cpu().numpy()

    np.save(os.path.join(args.outdir, f"{pid}.npy"), contact_map)
