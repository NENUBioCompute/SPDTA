import pandas as pd
import numpy as np
import os
import json, pickle
from collections import OrderedDict
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from utils import *

def smi2feats(smi, max_smi_len=102):
    smi = smi.replace(' ', '')

    X = [START_TOKEN]
    for ch in smi[: max_smi_len - 2]:
        X.append(SMI_CHAR_DICT[ch])
    X.append(END_TOKEN)
    X += [PAD_TOKEN] * (max_smi_len - len(X))
    X = np.array(X).astype(np.int64)
    return X

SMI_CHAR_DICT = {"(": 1, ".": 2, "0": 3, "2": 4, "4": 5, "6": 6, "8": 7, "@": 8,
                "B": 9, "D": 10, "F": 11, "H": 12, "L": 13, "N": 14, "P": 15, "R": 16,
                "T": 17, "V": 18, "Z": 19, "\\": 20, "b": 21, "d": 22, "f": 23, "h": 24,
                "l": 25, "n": 26, "r": 27, "t": 28, "#": 29, "%": 30, ")": 31, "+": 32,
                "-": 33, "/": 34, "1": 35, "3": 36, "5": 37, "7": 38, "9": 39, "=": 40,
                "A": 41, "C": 42, "E": 43, "G": 44, "I": 45, "K": 46, "M": 47, "O": 48,
                "S": 49, "U": 50, "W": 51, "Y": 52, "[": 53, "]": 54, "a": 55, "c": 56,
                "e": 57, "g": 58, "i": 59, "m": 60, "o": 61, "s": 62, "u": 63, "y": 64,
                ":": 65, "*": 66, "|": 67,
                }
assert np.all(np.array(sorted(list(SMI_CHAR_DICT.values()))) == np.arange(1, len(SMI_CHAR_DICT) + 1))
PAD_TOKEN = 0
START_TOKEN = len(SMI_CHAR_DICT) + 1
END_TOKEN = START_TOKEN + 1
assert PAD_TOKEN not in SMI_CHAR_DICT and START_TOKEN not in SMI_CHAR_DICT.values() and END_TOKEN not in SMI_CHAR_DICT
SMI_CHAR_SET_LEN = len(SMI_CHAR_DICT) + 3  # + (PADDING, START, END)



def seq_cat(prot, max_seq_len):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict[ch]
    return x

def struc_cat(prot_struc, max_len):
    x = np.zeros(max_len)
    for i,ch in enumerate(prot_struc[:max_len]):
        x[i] = struc_dict[ch]
    return x

all_prots = []
datasets = ['kiba', 'davis']
for dataset in datasets:
    print('convert data from DeepDTA for ', dataset)
    fpath = 'data/' + dataset + '/'
    train_fold = json.load(open(fpath + "folds/train_fold_setting1.txt"))
    train_fold = [ee for e in train_fold for ee in e]
    valid_fold = json.load(open(fpath + "folds/test_fold_setting1.txt"))
    ligands = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)
    proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)
    prot_struc = json.load(open(fpath + 'protein_structure.txt'), object_pairs_hook=OrderedDict)
    affinity = pickle.load(open(fpath + "Y", "rb"), encoding='latin1')
    drugs = []
    prots = []
    prot_struc_list = []
    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
        drugs.append(lg)
    for t in proteins.keys():
        prots.append(proteins[t])
        prot_struc_list.append(prot_struc[t])


    if dataset == 'davis':
        affinity = [-np.log10(y / 1e9) for y in affinity]
    affinity = np.asarray(affinity)
    opts = ['train', 'test']
    for opt in opts:
        rows, cols = np.where(np.isnan(affinity) == False)
        if opt == 'train':
            rows, cols = rows[train_fold], cols[train_fold]
        elif opt == 'test':
            rows, cols = rows[valid_fold], cols[valid_fold]
        with open('data/' + dataset + '_' + opt + '.csv', 'w') as f:
            f.write('compound_iso_smiles,target_sequence,target_structure,affinity\n')
            for pair_ind in range(len(rows)):
                ls = []
                ls += [drugs[rows[pair_ind]]]
                ls += [prots[cols[pair_ind]]]
                ls += [prot_struc_list[cols[pair_ind]]]
                ls += [affinity[rows[pair_ind], cols[pair_ind]]]
                f.write(','.join(map(str, ls)) + '\n')
    print('\ndataset:', dataset)
    print('train_fold:', len(train_fold))
    print('test_fold:', len(valid_fold))
    print('len(set(drugs)),len(set(prots)):', len(set(drugs)), len(set(prots)))
    all_prots += list(set(prots))

seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 1000

struc_dict = {"C": 1,"H": 2,"E": 3}


compound_iso_smiles = []
for dt_name in ['kiba', 'davis']:
    opts = ['train', 'test']
    for opt in opts:
        df = pd.read_csv('data/' + dt_name + '_' + opt + '.csv')
        compound_iso_smiles += list(df['compound_iso_smiles'])
compound_iso_smiles = set(compound_iso_smiles)
morgan = {}
for smile in compound_iso_smiles:
    mol = Chem.MolFromSmiles(smile)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
    npfp = np.array(list(fp.ToBitString())).astype('int8')
    morgan[smile] = npfp


datasets = ['davis', 'kiba']
# convert to PyTorch data format
for idx, dataset in enumerate(datasets):
    if idx == 0:
        max_len = 1200
    else:
        max_len = 1000
    processed_data_file = 'data/processed/' + dataset + '.pt'
    if (not os.path.isfile(processed_data_file)):
        df = pd.read_csv('data/' + dataset + '_train.csv')
        train_drugs, train_prots, train_prots_struc,train_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(df['target_structure']),list(
            df['affinity'])
        XT = [seq_cat(t, max_len) for t in train_prots]
        XD = [smi2feats(smi) for smi in train_drugs]
        struc = [struc_cat(t, max_len) for t in train_prots_struc]
        train_drugs, train_prots, train_prots_struc, train_Y, train_smiles_word = np.asarray(train_drugs), np.asarray(XT), np.asarray(struc) ,np.asarray(train_Y), np.asarray(XD)

        df = pd.read_csv('data/' + dataset + '_test.csv')
        test_drugs, test_prots, test_prots_struc,test_Y = list(df['compound_iso_smiles']), list(df['target_sequence']),list(df['target_structure']) ,list(
            df['affinity'])
        XT = [seq_cat(t, max_len) for t in test_prots]


        XD = [smi2feats(smi) for smi in test_drugs]
        struc = [struc_cat(t, max_len) for t in test_prots_struc]
        # print(struc)
        test_drugs, test_prots, test_prots_struc, test_Y, test_smiles_word = np.asarray(test_drugs), np.asarray(XT), np.asarray(struc),np.asarray(test_Y), np.asarray(XD)

        drugs = np.concatenate((train_drugs, test_drugs), axis=0)
        prots = np.concatenate((train_prots, test_prots), axis=0)
        Y = np.concatenate((train_Y, test_Y), axis=0)
        prots_struc = np.concatenate((train_prots_struc, test_prots_struc), axis=0)
        smiles_word = np.concatenate((train_smiles_word, test_smiles_word), axis=0)


        # make data PyTorch Geometric ready
        print('preparing ', dataset + '.pt in pytorch format!')
        data = TestbedDataset(root='data/', dataset=dataset, xd=drugs, xt=prots, y=Y, morgan = morgan,smiles_word=smiles_word, prots_struc=prots_struc)
    else:
        print(processed_data_file, ' are already created')

