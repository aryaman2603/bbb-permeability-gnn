import torch, torch_geometric, sklearn
from rdkit import Chem
print('PyTorch   :', torch.__version__)
print('PyG       :', torch_geometric.__version__)
print('sklearn   :', sklearn.__version__)
print('RDKit     : OK')
train = torch.load('data/processed/train.pt', weights_only=False)
val   = torch.load('data/processed/val.pt',   weights_only=False)
test  = torch.load('data/processed/test.pt',  weights_only=False)
print(f'Train: {len(train)} graphs | Val: {len(val)} | Test: {len(test)}')
print(f'Node dim : {train[0].x.shape[1]}  (should be 22)')
print(f'Edge dim : {train[0].edge_attr.shape[1]}  (should be 7)')
print(f'Has SMILES: {hasattr(train[0], chr(115)+chr(109)+chr(105)+chr(108)+chr(101)+chr(115))}')