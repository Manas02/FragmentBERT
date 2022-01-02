#!/usr/bin/env python
# coding: utf-8
# Author : Manas Mahale <manas.mahale@bcp.edu.in>
# Large Languge Models are Fragment Based Drug Designers

import random  
from rdkit import RDLogger, Chem
from rdkit.Chem import BRICS

import os
from transformers import BertForMaskedLM, pipeline, PreTrainedTokenizerFast


tokenizer = PreTrainedTokenizerFast.from_pretrained('./tokenizer/')
tokenizer.mask_token = "[MASK]"
tokenizer.unk_token = "[UNK]"
tokenizer.pad_token = "[PAD]"
tokenizer.sep_token = "[SEP]"
tokenizer.cls_token = "[CLS]"
model = BertForMaskedLM.from_pretrained(os.path.join('model/', "checkpoint-250"))

fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)


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

RDLogger.DisableLog('rdApp.*') 

def main(smi):
    d = list(BRICS.BRICSDecompose(Chem.MolFromSmiles(smi)))
    a = [mol_to_smiles(strip_dummy_atoms(mol_from_smiles(i))) for i in d]
    a.insert(random.randint(0, len(a)), '[MASK]')
    a = ' '.join(a)
    print(a)
    for prediction in fill_mask(a):
        print(prediction)

if __name__ == '__main__':
    smi = 'CCc1nc(CNc2cccc3cccnc23)cs1'
    
    main(smi)