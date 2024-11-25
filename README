# Protein Language Model Reliability Score

## Overview

This project implements a reliability scoring method for protein language models (pLMs), addressing the implicit biases these models inherit from their training data and architectures. By leveraging contact map differences between predicted and actual protein structures, we aim to create a supervised learning approach to efficiently assess prediction reliability for various pLMs.

The outcomes will enable researchers to evaluate prediction reliability for tasks such as protein function prediction or design.

This was a project within the course **Protein Prediction 2** at TUM (Technical University of Munich).

## Features
- Use of Jacobian matrix-derived contact maps for structure comparison.
- Creation of a supervised model to predict a reliability score.
- Compatibility with the ESM-2 protein language models with models to be added

## Tasks
1. **Dataset Processing**: Perform redundancy reduction and split into training, validation, and testing sets.
2. **Contact Map Computation**:
    - Compute Jacobian matrices and derive contact maps for each protein.
    - Calculate differences between predicted and actual structures of the proteins.
4. **Model Development**:
    - Train a supervised learning model to predict the reliability score based on a protein embedding.
    - Evaluate the model using suitable metrics.


## Citations
This project builds on the following works:
1. Zhang, Z. et al., Protein language models learn evolutionary statistics of interacting sequence motifs, *bioRxiv*, 2024.
2. Lin, Z. et al., Language models of protein sequences at the scale of evolution enable accurate structure prediction, *bioRxiv*, 2022.
3. Elnaggar, A. et al., ProtTrans: towards cracking the language of life’s code, *IEEE Transactions*, 2021.
4. Heinzinger, M. et al., ProstT5: Bilingual Language Model for Protein Sequence and Structure, *bioRxiv*, 2023.
