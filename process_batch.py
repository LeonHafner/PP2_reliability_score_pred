#!/usr/bin/env python3

import os
import argparse
from tqdm import tqdm
import torch
import numpy as np
import esm

from jacobian import *

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
    lines = [line.strip().split(';') for line in f.readlines() if line.strip()]

for pid, seq in tqdm(lines):
    jacobian = get_jacobian(seq, model, alphabet, device)
    
    contact_map = get_contact_map(jacobian, center=args.center, apc=args.apc, sym=args.sym)

    if isinstance(contact_map, torch.Tensor):
        contact_map = contact_map.cpu().numpy()

    np.save(os.path.join(args.outdir, f"{pid}.npy"), contact_map)
