"""Defines high-level objects for interfacing with raw SidechainNet data.

To utilize SCNDataset, pass scn_dataset=True to scn.load().

    >>> d = scn.load("debug", scn_dataset=True)
    >>> d
    SCNDataset(n=461)

SCNProteins may be iterated over or selected from the SCNDataset.

    >>> d["1HD1_1_A"]
    SCNProtein(1HD1_1_A, len=75, missing=0, split='train')

Available SCNProtein attributes include:
    * coords
    * angles
    * seq
    * unmodified_seq
    * mask
    * evolutionary
    * secondary_structure
    * resolution
    * is_modified
    * id
    * split

Other features:
    * add non-terminal hydrogens to a protein's coordinates with SCNProtein.add_hydrogens
    * visualize proteins with SCNProtein.to_3Dmol() 
    * write PDB files for proteins with SCNProtein.to_PDB()
"""
import sidechainnet
import sidechainnet.structure.HydrogenBuilder as hy
from sidechainnet import structure
from sidechainnet.structure.build_info import NUM_COORDS_PER_RES
from sidechainnet.utils.sequence import ONE_TO_THREE_LETTER_MAP


class SCNDataset(object):
    """A representation of a SidechainNet dataset."""

    def __init__(self, data) -> None:
        """Initialize a SCNDataset from underlying SidechainNet formatted dictionary."""
        super().__init__()
        # Determine available datasplits
        self.splits = []
        for split_name in ['train', 'valid', 'test']:
            for k in data.keys():
                if split_name in k:
                    self.splits.append(k)

        self.split_to_ids = {}
        self.ids_to_SCNProtein = {}
        self.idx_to_SCNProtein = {}

        # Create SCNProtein objects and add to data structure
        for split in self.splits:
            d = data[split]
            for i in range(len(d['crd'])):
            # for c, a, s, u, m, e, n, r, z, i in zip(d['crd'], d['ang'], d['seq'],
            #                                         d['ums'], d['msk'], d['evo'],
            #                                         d['sec'], d['res'], d['mod'],
            #                                         d['ids']):
                try:
                    self.split_to_ids[split].append(d['ids'][i])
                except KeyError:
                    self.split_to_ids[split] = [d['ids'][i]]

                if 'blens' in d:
                    bond_lengths = d['blens'][i]
                else:
                    bond_lengths = None

                p = SCNProtein(coordinates=d['crd'][i],
                               angles=d['ang'][i],
                               sequence=d['seq'][i],
                               unmodified_seq=d['ums'][i],
                               mask=d['msk'][i],
                               evolutionary=d['evo'][i],
                               secondary_structure=d['sec'][i],
                               resolution=d['res'][i],
                               is_modified=d['mod'][i],
                               id=d['ids'][i],
                               bond_lengths=bond_lengths,
                               split=split)
                self.ids_to_SCNProtein[d['ids'][i]] = p
                self.idx_to_SCNProtein[i] = p


    def get_protein_list_by_split_name(self, split_name):
        """Return list of SCNProtein objects belonging to str split_name."""
        return [p for p in self if p.split == split_name]

    def __getitem__(self, id):
        """Retrieve a protein by index or ID (name, e.g. '1A9U_1_A')."""
        if isinstance(id, str):
            return self.ids_to_SCNProtein[id]
        elif isinstance(id, slice):
            step = 1 if not id.step else id.step
            stop = len(self) if not id.stop else id.stop
            start = 0 if not id.start else id.start
            stop = len(self) + stop if stop < 0 else stop
            start = len(self) + start if start < 0 else start
            return [self.idx_to_SCNProtein[i] for i in range(start, stop, step)]
        else:
            return self.idx_to_SCNProtein[id]

    def __len__(self):
        """Return number of proteins in the dataset."""
        return len(self.idx_to_SCNProtein)

    def __iter__(self):
        """Iterate over SCNProtein objects."""
        yield from self.ids_to_SCNProtein.values()

    def __repr__(self) -> str:
        """Represent SCNDataset as a string."""
        return f"SCNDataset(n={len(self)})"

    def filter_ids(self, to_keep):
        """Remove proteins whose IDs are not included in list to_keep."""
        to_delete = []
        for pnid in self.ids_to_SCNProtein.keys():
            if pnid not in to_keep:
                to_delete.append(pnid)
        for pnid in to_delete:
            p = self.ids_to_SCNProtein[pnid]
            self.split_to_ids[p.split].remove(pnid)
            del self.ids_to_SCNProtein[pnid]
        self.idx_to_SCNProtein = {}
        for i, protein in enumerate(self):
            self.idx_to_SCNProtein[i] = protein


class SCNProtein(object):
    """Represent one protein in SidechainNet. Created programmatically by SCNDataset."""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.coords = kwargs['coordinates']
        self.angles = kwargs['angles']
        if kwargs['bond_lengths'] is not None:
            self.bond_lengths = kwargs['bond_lengths']
        self.seq = kwargs['sequence']
        self.unmodified_seq = kwargs['unmodified_seq']
        self.mask = kwargs['mask']
        self.evolutionary = kwargs['evolutionary']
        self.secondary_structure = kwargs['secondary_structure']
        self.resolution = kwargs['resolution']
        self.is_modified = kwargs['is_modified']
        self.id = kwargs['id']
        self.split = kwargs['split']
        self.sb = None
        self.atoms_per_res = NUM_COORDS_PER_RES
        self.hcoords = None  # Coordinates with hydrogen atoms
        self._has_hydrogens = False

    def __len__(self):
        """Return length of protein sequence."""
        return len(self.seq)

    def to_3Dmol(self):
        """Return an interactive visualization of the protein with py3DMol."""
        if self.sb is None:
            if self._has_hydrogens:
                self.sb = sidechainnet.StructureBuilder(self.seq, self.hcoords)
            else:
                self.sb = sidechainnet.StructureBuilder(self.seq, self.coords)
        return self.sb.to_3Dmol()

    def to_pdb(self, path, title=None):
        """Save structure to path as a PDB file."""
        if not title:
            title = self.id
        if self.sb is None:
            if self._has_hydrogens:
                self.sb = sidechainnet.StructureBuilder(self.seq, self.hcoords)
            else:
                self.sb = sidechainnet.StructureBuilder(self.seq, self.coords)
        return self.sb.to_pdb(path, title)

    @property
    def num_missing(self):
        """Return number of missing residues."""
        return self.mask.count("-")

    @property
    def seq3(self):
        """Return 3-letter amino acid sequence for the protein."""
        return " ".join([ONE_TO_THREE_LETTER_MAP[c] for c in self.seq])

    def add_hydrogens(self, coords=None, build_from_angles=False):
        """Add non-terminal hydrogens to coordinates. Shapes (14L x 3) -> (24L x 3).

        Adds hydrogen atoms to the protein structure coordinate representation. This
        function essentially converts the coordinate maping schema from one that only
        includes heavy atoms (i.e., NUM_COORDS_PER_RES = 14 atoms per residue) to one that
        includes hydrogen atoms (i.e., NUM_COORDS_PER_RES_W_HYDROGENS = 24 atoms per
        residue). For simplicity, N-terminal hydrogens (H2 and H3) and the terminal oxygen
        (OXT) are not explicitly included in the atom mapping. Rather, they are stored in
        StructureBuilder.terminal_atoms, a dictionary mapping terminal atom names to their
        coordinates.

        See
        sidechainnet.structure.HydrogenBuilder for more details.

        Args:
            coords (np.ndarray, torch.tensor, optional): A set of heavy-atom coordinates
                which can be provided to override the current atom coordinates before
                adding hydrogens. Defaults to None.
            build_from_angles (bool, optional): If True, rebuild heavy-atom coordinates
                from internal angle representaion before adding hydrogens. Defaults to
                False.
        """
        # Initialize a structure builder with heavy-atom coordinates
        if build_from_angles:
            self.sb = structure.StructureBuilder(self.seq, ang=self.angles)
            self.sb.build()
        elif coords is not None:
            self.sb = structure.StructureBuilder(self.seq, crd=coords)
        else:
            self.sb = structure.StructureBuilder(self.seq, crd=self.coords)

        self.sb.add_hydrogens()
        self.hcoords = self.sb.coords
        self._has_hydrogens = True
        self.atoms_per_res = hy.NUM_COORDS_PER_RES_W_HYDROGENS
        return self.hcoords

    def __repr__(self) -> str:
        """Represent an SCNProtein as a string."""
        return (f"SCNProtein({self.id}, len={len(self)}, missing={self.num_missing}, "
                f"split='{self.split}')")
