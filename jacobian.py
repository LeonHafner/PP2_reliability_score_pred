#!/usr/bin/env python3

import torch


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
        jacobian = jacobian - torch.mean(jacobian, dim=dim, keepdim=True)
    
    contact_map = torch.sqrt(torch.sum(jacobian ** 2, dim=(1, 3))).fill_diagonal_(0)

    if apc:
        contact_map = apply_apc(contact_map)
    
    if sym:
        contact_map = (contact_map + contact_map.T) / 2
    
    return contact_map


def apply_apc(contact_map):
    mean_row = contact_map.mean(dim=0, keepdim=True)
    mean_col = contact_map.mean(dim=1, keepdim=True)
    mean_total = contact_map.mean()
    apc = (mean_row * mean_col) / mean_total
    return contact_map - apc
