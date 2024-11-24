# Usage
# pip install tqdm scipy numpy jax h5py
# python process_batch train_X.csv output_folder/ <apc>
# train_X.csvs are on Google Drive


import time
import os 
import torch 
import json 
from tqdm import tqdm 
import numpy as np
from scipy.special import softmax
from utils import *
import random 
import matplotlib.pyplot as plt
import sys


def get_categorical_jacobian(seq, model, alphabet, device):
  # ∂in/∂out
  x,ln = alphabet.get_batch_converter()([("seq",seq)])[-1],len(seq)
  with torch.no_grad():
    f = lambda x: model(x)["logits"][...,1:(ln+1),4:24].cpu().numpy()
    fx = f(x.to(device))[0]
    x = torch.tile(x,[20,1]).to(device)
    fx_h = np.zeros((ln,20,ln,20))
    for n in range(ln): # for each position
      x_h = torch.clone(x)
      x_h[:,n+1] = torch.arange(4,24) # mutate to all 20 aa
      fx_h[n] = f(x_h)
    return fx_h - fx

def get_contacts(x, symm=True, center=True, rm=1, apc=False):
  # convert jacobian (L,A,L,A) to contact map (L,L)
  j = x.copy()
  if center:
    for i in range(4): j -= j.mean(i,keepdims=True)
  j_fn = np.sqrt(np.square(j).sum((1,3)))
  np.fill_diagonal(j_fn,0)
    
  if apc:
      j_fn_corrected = do_apc(j_fn, rm=rm)
      if symm:
        j_fn_corrected = (j_fn_corrected + j_fn_corrected.T)/2
      return j_fn_corrected
  else:
      return j_fn

def do_apc(x, rm=1):
  '''given matrix do apc correction'''
  # trying to remove different number of components
  # rm=0 remove none
  # rm=1 apc
  x = np.copy(x)
  if rm == 0:
    return x
  elif rm == 1:
    a1 = x.sum(0,keepdims=True)
    a2 = x.sum(1,keepdims=True)
    y = x - (a1*a2)/x.sum()
  else:
    # decompose matrix, rm largest(s) eigenvectors
    u,s,v = np.linalg.svd(x)
    y = s[rm:] * u[:,rm:] @ v[rm:,:]
  np.fill_diagonal(y,0)
  return y

def main():
    # Usage
    # pip install tqdm scipy numpy jax h5py
    # python process_batch train_X.csv output_folder/ <apc>
    # apc argument is optional
    
    filepath_to_protein_csv = sys.argv[1]
    out_folder = sys.argv[2]
    apc_on = True if sys.argv[3] == "apc" else False
    print("Start Processing: ", filepath_to_protein_csv)
    print("Outfolder: ", out_folder)
    print(f"APC?: {apc_on}")
    
    
    print("\n\nLoading model, ... could take a while \n\n ")
    model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    model = model.to(device)
    model = model.eval()
    print("Model loading successful")
    
    # Load csv part
    with open(filepath_to_protein_csv, "r") as rf:
        lines = rf.read().split("\n")
        print(f"Found {len(lines)-1} Proteins")
        for line in tqdm(lines, desc="Processing proteins:"):
            if ";" in line:
                pid, sequence = line.split(";")
                cjm1 = get_categorical_jacobian(sequence, model, alphabet, device)
                contact_map = get_contacts(cjm1, model, alphabet, apc=apc_on)
                out_path = os.path.join(out_folder, pid+".npy")
                np.save(out_path, contact_map)

if __name__ == "__main__":
    main()