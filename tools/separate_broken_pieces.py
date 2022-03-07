'''
This code separate pieces in validation and test set proteins into multiple entries to resolve missing residue problem
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

def find_slices(mask, min_allowed_len=50):
    mask_int = np.array([m == '+' for m in mask]).astype(int)
    change_positions = np.where(mask_int[1:] - mask_int[:-1])[0]
    positions = [0] + [i+1 for i in change_positions] + [-1]
    starting_label=mask_int[0]
    slices = []
    if starting_label == 0:
        slice_idx = 1
    else:
        slice_idx = 0
    while slice_idx < len(positions) - 1:
        if positions[slice_idx + 1] - positions[slice_idx] >= min_allowed_len:
            slices.append(slice(positions[slice_idx], positions[slice_idx + 1]))
        slice_idx += 2
    return slices


for key in data:
    if 'valid' in key or 'test' in key:
        # we only need to process validation and test
        print(key)
        new_data = {k: [] for k in data[key]}
        for idx, mask in enumerate(data[key]['msk']):
            if '-' not in mask:
                for item in new_data:
                    new_data[item].append(data[key][item][idx])
            else:
                slices = find_slices(mask)
                ums_list = data[key]['ums'][idx].split()
                for sl in slices:
                    new_data['ids'].append(data[key]['ids'][idx] + f"_{sl.start}-{sl.stop}")
                    new_data['res'].append(data[key]['res'][idx])
                    new_data['crd'].append(data[key]['crd'][idx][sl.start * 14: sl.stop * 14])
                    new_data['ums'].append(" ".join(ums_list[sl]))
                    for item in new_data:
                        if item not in ['ids', 'res', 'crd', 'ums']:
                            new_data[item].append(data[key][item][idx][sl])
        data[key] = new_data

with open(pkl_file, 'wb') as f:
    pickle.dump(data, f)

print("Finish!")