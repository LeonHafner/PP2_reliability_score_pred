#!/usr/bin/env python3

"""
This script reads a CSV file containing protein sequences and their identifiers, 
uses a pretrained ESM model to generate sequence embeddings, and saves the 
embeddings as a parquet file. 

The input CSV file should contain lines of comma-separated values: the first value 
is the sequence identifier (label) and the second value is the protein sequence. 
Each sequence is then processed by the ESM model, and the resulting embeddings are 
stored in a parquet file with identifiers as the index. The CSV file is supposed to have no header.

Example usage:
    python3 get_protein_embeddings.py --input test_batch.csv --output embeddings.parquet --batch_size 32
"""

import argparse
import pandas as pd
import torch
import esm
from tqdm import tqdm


parser = argparse.ArgumentParser(description="Generate ESM embeddings from protein sequences.")
parser.add_argument("--input", type=str, required=True, help="Path to input CSV file containing sequences.")
parser.add_argument("--output", type=str, required=True, help="Path to output parquet file.")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for ESM model inference.")
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on {device}")

with open(args.input, "r") as f:
    data = [tuple(line.strip().split(',')) for line in f.readlines() if line.strip()]

print("Loading model...")
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model = model.eval()
model = model.to(device)
batch_converter = alphabet.get_batch_converter()

sequence_representations = []
sequence_indices = []
for i in tqdm(range(0, len(data), args.batch_size)):
    batch_data = data[i:i+args.batch_size]
    batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)

    batch_tokens = batch_tokens.to(device)
    batch_lengths = (batch_tokens != alphabet.padding_idx).sum(1)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)

    token_representations = results["representations"][33]

    for j, token_len in enumerate(batch_lengths):
        sequence_representations.append(token_representations[j, 1:token_len-1].mean(0))
        sequence_indices.append(batch_labels[j])

embedding_df = pd.DataFrame([t.cpu().numpy() for t in sequence_representations], columns=[f"{i}" for i in range(1280)])
embedding_df.index = sequence_indices
print(embedding_df.shape)

embedding_df.to_parquet(args.output, index=True)