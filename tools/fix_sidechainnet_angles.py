'''
Some of the bond angle data in sidechainnet are not correct, 
which usually are bond angles of CA-C-N and C-N-CA for a residue with the next residue missing

This code set all CA-C-N and C-N-CA bond angles to 0 if the next residue is missing
'''

import os
import pickle
import numpy as np
import shutil
from tqdm import tqdm

pkl_file = '../../sidechainnet_data/sidechainnet_casp12_30.pkl' # the pickle file for processing

if os.path.exists(pkl_file + '.backup'):
    shutil.copy(pkl_file + '.backup', pkl_file)
else:
    shutil.copy(pkl_file, pkl_file + '.backup')

with open(pkl_file, 'rb') as f:
    data = pickle.load(f)

for key in data:
    if 'train' in key or 'valid' in key or 'test' in key:
        # this is data record field
        print(key)
        for idx, mask in enumerate(data[key]['msk']):
            neg_positions = [i for i in range(len(mask)) if mask[i] == '-']
            for pos in neg_positions:
                if pos != 0:
                    data[key]['ang'][idx][pos, 4:6] = 0

with open(pkl_file, 'wb') as f:
    pickle.dump(data, f)

print("Finish!")