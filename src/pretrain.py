#!/usr/bin/env python
# coding: utf-8
# Author : Manas Mahale <manas.mahale@bcp.edu.in>
# Large Languge Models are Fragment Based Drug Designers

from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit import RDLogger           


def mol_to_smiles(mol):
        return Chem.MolToSmiles(mol, isomericSmiles=True)

def mol_from_smiles(smi):
    return Chem.MolFromSmiles(smi)

dummy = Chem.MolFromSmiles('[*]')

def strip_dummy_atoms(mol):
    hydrogen = mol_from_smiles('[H]')
    mols = Chem.ReplaceSubstructs(mol, dummy, hydrogen, replaceAll=True)
    mol = Chem.RemoveHs(mols[0])
    return mol

with open('smiles/canonical_train.txt') as f:
    data = [i.strip() for i in f.readlines()]

RDLogger.DisableLog('rdApp.*') 

with open('smiles/canonical_train_scaffold.txt', 'w') as f:
    for i in tqdm(data):
        f.write(' '.join([mol_to_smiles(strip_dummy_atoms(mol_from_smiles(j))) for j in list(BRICS.BRICSDecompose(mol_from_smiles(i)))]) + '\n')
