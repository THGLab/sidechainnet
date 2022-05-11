'''
Prepare the sidechainnet data in IDPConformerGenerator format (Input torsions are omega_{i-1}, phi_i, psi_i, bond lengths are N-CA, CA-C, C-N{i+1},
bond angles are C{i-1}-N-CA, N-CA-C, CA-C-N{i+1})
'''

import os
import shutil
import pickle

import numpy as np

pkl_file = '../../sidechainnet_data/sidechainnet_debug.pkl' # the pickle file for processing

if os.path.exists(pkl_file + '.original'):
    shutil.copy(pkl_file + '.original', pkl_file)
else:
    shutil.copy(pkl_file, pkl_file + '.original')

with open(pkl_file, 'rb') as f:
    data = pickle.load(f)

for key in data:
    if 'train' in key or 'valid' in key or 'test' in key:
        # this is data record field
        print(key)
        data_record = data[key]
        
        for idx, item in enumerate(data_record["ang"]):
            omegas = item[:, 2]
            new_omegas = np.concatenate([[0], omegas[:-1]])

            theta3s = item[:, 5]
            new_theta3s = np.concatenate([[0], theta3s[:-1]])

            data_record["ang"][idx][:, 2] = new_omegas
            data_record["ang"][idx][:, 5] = new_theta3s
        

with open(pkl_file, 'wb') as f:
    pickle.dump(data, f)

print("Finish!")