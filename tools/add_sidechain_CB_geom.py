import os
import pickle
import numpy as np
import shutil
from tqdm import tqdm

pkl_file = '../../sidechainnet_data/sidechainnet_casp12_70.pkl' # the pickle file for processing

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

        # process CA-CB bond lengths and N-CA-CB bond angles
        bond_lens = []
        bond_angs = []
        for idx, coord in tqdm(enumerate(data[key]['crd'])):
            n_atoms = len(data[key]['seq'][idx])
            coord = coord.reshape((-1, 14, 3))
            assert len(coord) == n_atoms
            # create the bond length array first
            bl = np.zeros((n_atoms, 1))
            ang = np.zeros((n_atoms, 1))
            n_coords = coord[:, 0]
            ca_coords = coord[:, 1]
            cb_coords = coord[:, 4]

            # valid atoms are those atoms whose coordinates are not all zero
            n_valid = np.linalg.norm(n_coords, axis=-1) != 0
            ca_valid = np.linalg.norm(ca_coords, axis=-1) != 0
            cb_valid = np.linalg.norm(cb_coords, axis=-1) != 0

            ca_n_vectors = n_coords - ca_coords
            ca_cb_vectors = cb_coords - ca_coords
            ca_n_dist = np.linalg.norm(ca_n_vectors, axis=-1)
            ca_cb_dist = np.linalg.norm(ca_cb_vectors, axis=-1)
            n_ca_cb_angles = np.arccos((ca_n_vectors.reshape((-1,1,3)) @ \
                ca_cb_vectors.reshape((-1,3,1))).flatten() / (ca_n_dist * ca_cb_dist))

            bl[:, 0] = ca_cb_dist * (ca_valid & cb_valid).astype(float) 
            ang[:, 0] = np.nan_to_num(n_ca_cb_angles) * (n_valid & ca_valid & cb_valid).astype(float)

            bond_lens.append(bl)
            bond_angs.append(ang)
        data[key]['sc-blens'] = bond_lens
        data[key]['sc-angs'] = bond_angs



with open(pkl_file, 'wb') as f:
    pickle.dump(data, f)

print("Finish!")