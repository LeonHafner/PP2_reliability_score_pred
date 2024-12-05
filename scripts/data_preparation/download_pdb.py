#!/usr/bin/env python3

"""
This script downloads PDB files from a list of PDB IDs provided in a CSV file.

The script accepts the following arguments:
1. --csv_file: Path to the input CSV file containing PDB IDs in a single column.
2. --output: Path to the directory where the downloaded PDB files will be saved.

The script processes each PDB ID and downloads the corresponding PDB file from the PDB website. 
Files are saved with the format '<PDB_ID>-<CHAIN>.pdb' in the specified output directory. If a file already exists,
it is skipped. Errors during download (e.g., network issues or invalid IDs) are reported.

Example Usage:
    python script.py --csv_file pdb_list.csv --output ./pdb_files

Where `pdb_list.csv` contains entries like:
    1abc-A
    2xyz-B
    3def-C

And `./pdb_files` is the directory where files like '1abc-A.pdb' will be stored.

Author: Leon Hafner
"""


import os
import argparse
from tqdm import tqdm
import requests

parser = argparse.ArgumentParser(description="Download PDB files from a list in a CSV file.")
parser.add_argument("--csv_file", type=str, help="Path to the input CSV file (with only one column) containing PDB IDs", required=True)
parser.add_argument("--output", type=str, help="Path to the folder where PDB files will be saved.", required=True)

args = parser.parse_args()

csv_file = args.csv_file
output = args.output

os.makedirs(output, exist_ok=True)

with open(csv_file, "r") as file:
    lines = file.readlines()

for line in tqdm(lines):
    pdb_entry = line.strip()

    if not pdb_entry:
        continue

    pdb_id = pdb_entry.split('-')[0].lower()
    chain = pdb_entry.split('-')[1].upper()

    pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    output_file = os.path.join(output, f"{pdb_id}-{chain}.pdb")

    if os.path.exists(output_file):
        continue

    try:
        response = requests.get(pdb_url)
        response.raise_for_status()  # Raise an error for bad HTTP response
        with open(output_file, 'w') as pdb_file:
            pdb_file.write(response.text)
    except requests.exceptions.RequestException as e:
                print(f"Failed to download {pdb_id}-{chain}: {e}")
print("Finished download!")
