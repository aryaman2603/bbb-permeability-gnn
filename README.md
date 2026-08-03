# BBB Permeability Prediction with Graph Neural Networks

Predicting Blood-Brain Barrier (BBB) permeability of drug-like molecules using Graph Attention Networks (GAT), Graph Isomorphism Networks (GIN), and a Random Forest baseline -- trained on the BBBP benchmark dataset.

---

## Table of Contents

- [Overview](#overview)
- [Background](#background)
- [Models](#models)
- [Results](#results)

---

## Overview

The **Blood-Brain Barrier (BBB)** is a highly selective semipermeable membrane that separates circulating blood from the brain. Whether a drug molecule can cross this barrier is a critical property in CNS drug discovery -- a molecule that cannot penetrate the BBB cannot treat brain diseases.

This project frames BBB permeability as a **binary graph classification task**, treating each molecule as a molecular graph where:
- **Nodes** = atoms (with 22-dimensional feature vectors)
- **Edges** = bonds (with 7-dimensional feature vectors)
- **Label** = BBB+ (permeable) or BBB- (non-permeable)

Two GNN architectures (GAT and GIN) are compared against a classical Random Forest baseline using Morgan fingerprints.

---

## Background

### Why GNNs for Molecular Property Prediction?

Traditional cheminformatics relies on hand-crafted molecular descriptors or fingerprints (like ECFP4). While powerful, these representations lose the **explicit graph topology** of molecules. GNNs operate directly on the molecular graph, enabling them to learn task-specific structural patterns through end-to-end training.

### Dataset: BBBP

The [Blood-Brain Barrier Penetration (BBBP)](https://moleculenet.org/datasets-1) dataset from MoleculeNet contains **~2,050 molecules** with binary labels indicating BBB permeability. The dataset has a notable class imbalance (~75% BBB+), which is handled explicitly during training.

---

## Models

### Graph Attention Network (GAT)

GAT replaces the fixed, normalized neighbor aggregation of standard GCNs with a **learned attention mechanism**. For each atom, it computes an attention score for every neighbor, reflecting how important that neighbor is to the central atom's representation.

**Architecture:**

| Layer  | Operation                          | Output Dim |
|--------|------------------------------------|------------|
| Conv 1 | GATConv (4 heads, concat)          | 256        |
| Conv 2 | GATConv (4 heads, concat)          | 256        |
| Conv 3 | GATConv (4 heads, avg)             | 64         |
| Pool   | Global Mean Pool                   | 64         |
| MLP    | Linear -> ReLU -> Dropout -> Linear | 1 (logit)  |

---

### Graph Isomorphism Network (GIN)

GIN is theoretically the most expressive standard GNN -- provably as powerful as the Weisfeiler-Lehman (WL) graph isomorphism test. It uses a **learnable MLP** and a **learnable epsilon parameter**, with SUM aggregation (unlike MEAN-based methods that can conflate structurally different graphs).

A custom `GINConvWithEdge` layer projects bond features into node space and injects them into neighbor messages -- a lightweight but effective approach to handle edge attributes that the standard GINConv ignores.

**Architecture:**

| Layer      | Operation                              | Output Dim |
|------------|----------------------------------------|------------|
| Conv 1-3   | GINConvWithEdge (MLP + BatchNorm)      | 64         |
| Pool       | Mean Pool + Max Pool (concatenated)    | 128        |
| MLP        | Linear -> ReLU -> Dropout -> Linear     | 1 (logit)  |

---

### Random Forest Baseline (Morgan Fingerprints)

A classical ML baseline using 2048-bit ECFP4 (Morgan radius=2) fingerprints as input to a Random Forest classifier. This model does **not** use graph structure -- it serves to answer the key question: does graph structure add value beyond fingerprints?

- `n_estimators=500`, `max_features='sqrt'`, `class_weight='balanced'`
- Out-of-bag scoring for free validation estimates

---

## Results

All models are evaluated on the same frozen **scaffold-split test set** (15% of data, ~306 molecules). Scaffold splitting ensures no molecular scaffold appears in both train and test, providing a more realistic estimate of generalization.

### Test Set Performance

| Model          | ROC-AUC    | Accuracy   | Precision  | Recall     | F1         |
|----------------|------------|------------|------------|------------|------------|
| GIN            | **0.9025** | 0.843      | 0.919      | 0.872      | 0.895      |
| Random Forest  | 0.8903     | **0.873**  | 0.885      | **0.957**  | **0.920**  |
| GAT            | 0.8695     | 0.755      | **0.939**  | 0.726      | 0.819      |

Key findings:
- **GIN achieves the best ROC-AUC (0.9025)**, demonstrating that graph structure captures meaningful molecular patterns.
- **Random Forest** delivers competitive performance (0.890 AUC) with the highest recall, showing that ECFP4 fingerprints remain a strong baseline.
- **GAT** shows high precision but lower recall -- it is conservative in predicting BBB+ but misses more true positives.
- The GNN vs. RF comparison suggests molecular graphs provide complementary structural information beyond circular fingerprints.

