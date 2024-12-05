#!/usr/bin/env python3
"""
This script reads a list of protein entries from an input file and splits them into smaller CSV files.
Each output file will contain a specified number of proteins, allowing for easier handling of large datasets.
The resulting files will be placed in a designated output directory and named with a user-defined prefix,
followed by an incremented, zero-padded index (e.g., prefix_01.csv, prefix_02.csv).

Args:
    -i, --input (str): Path to the input file containing protein entries.
    -o, --output_dir (str): Path to the directory where the chunked files will be saved.
    -p, --prefix (str): Prefix used in naming the output files.
    -c, --chunks (int, optional): Number of protein entries per output file (default: 500).

Author: Leon Hafner
"""

import argparse
import os

parser = argparse.ArgumentParser(description='Split proteins into chunks.')
parser.add_argument('-i', '--input', required=True, help='Input file containing proteins.')
parser.add_argument('-o', '--output_dir', required=True, help='Output directory.')
parser.add_argument('-p', '--prefix', required=True, help='Files will have the format "{prefix}_{i}.csv".')
parser.add_argument('-c', '--chunks', type=int, default=500, help='Number of proteins per file (default: 500).')
args = parser.parse_args()

with open(args.input, 'r') as f:
    lines = f.readlines()

chunk_size = args.chunks
total_chunks = (len(lines) + chunk_size - 1) // chunk_size

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# Determine the number of digits needed for zero-padding based on total_chunks
num_digits = len(str(total_chunks))

for i in range(total_chunks):
    start_index = i * chunk_size
    end_index = start_index + chunk_size
    chunk = lines[start_index:end_index]
    output_file = os.path.join(args.output_dir, f'{args.prefix}_{i+1:0{num_digits}d}.csv')
    with open(output_file, 'w') as f:
        f.writelines(chunk)
    print(f'Created {output_file} with {len(chunk)} proteins.')
