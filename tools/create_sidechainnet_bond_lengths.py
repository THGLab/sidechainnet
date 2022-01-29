import os
import pickle
import numpy as np
import shutil
from tqdm import tqdm

pkl_file = 'sidechainnet_data/sidechainnet_casp12_30.pkl' # the pickle file for processing

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
        bond_lens = []
        for idx, coord in tqdm(enumerate(data[key]['crd'])):
            n_atoms = len(data[key]['seq'][idx])
            coord = coord.reshape((-1, 14, 3))
            assert len(coord) == n_atoms
            # create the bond length array first
            bl = np.zeros((n_atoms, 3))
            n_coords = coord[:, 0]
            ca_coords = coord[:, 1]
            c_coords = coord[:, 2]

            # valid atoms are those atoms whose coordinates are not all zero
            n_valid = np.linalg.norm(n_coords, axis=-1) != 0
            ca_valid = np.linalg.norm(ca_coords, axis=-1) != 0
            c_valid = np.linalg.norm(c_coords, axis=-1) != 0

            bl[:, 0] = np.linalg.norm((ca_coords - n_coords), axis=-1) * (ca_valid & n_valid).astype(float)  # n-ca bond lengths
            bl[:, 1] = np.linalg.norm((c_coords - ca_coords), axis=-1) * (c_valid & ca_valid).astype(float)  # ca-c bond lengths
            bl[:-1, 2] = np.linalg.norm((n_coords[1:] - c_coords[:-1]), axis=-1) * (n_valid[1:] & c_valid[:-1]).astype(float)  # c-n+1 bond lengths
            bl[:-1, 2] = bl[:-1, 2] * (bl[:-1, 2] < 2.4).astype(float)  # a filter for broken chains
            bond_lens.append(bl)
        data[key]['blens'] = bond_lens

with open(pkl_file, 'wb') as f:
    pickle.dump(data, f)

print("Finish!")