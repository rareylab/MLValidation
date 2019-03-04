import rdkit.Chem
import rdkit.Chem.AllChem
import rdkit.Geometry
import operator
import os
import argparse
import queue
import pandas as pd
import numpy as np


class TypePolicy:
    """
    Policy to handle atom types
    """
    all_atom_types = {}

    def __init__(self):
        self.t = None

    def discretize(self, atom_type):
        assert atom_type in TypePolicy.all_atom_types
        return TypePolicy.all_atom_types[atom_type]

    def invalid(self):
        return len(TypePolicy.all_atom_types)

    def set_molecule(self, m):
        n = m.GetNumAtoms()
        self.t = [None for i in range(n)]
        for idx in range(n):
            atom = m.GetAtomWithIdx(idx)
            atom_type = atom.GetSymbol()
            self.t[idx] = atom_type

    def get(self, idx):
        assert self.t is not None
        return self.t[idx]


class ChargePolicy:
    """
    Policy to handle charges.
    """
    def __init__(self):
        self.c = None
        self.charge_size = 0.05
        self.min_charge = -1.0
        self.max_charge = 1.0
        self.nof_charge_bins = int((self.max_charge - self.min_charge) / self.charge_size + 1)

    def discretize(self, chrg):
        assert chrg <= self.max_charge
        assert chrg >= self.min_charge
        return int((chrg - self.min_charge) / self.charge_size)

    def invalid(self):
        return self.nof_charge_bins

    def set_molecule(self, m):
        n = m.GetNumAtoms()
        self.c = np.zeros(n)
        self.c.fill(np.inf)
        for idx in range(n):
            atom = m.GetAtomWithIdx(idx)
            if atom.HasProp('_TriposPartialCharge'):
                partial_charge = float(atom.GetProp("_TriposPartialCharge"))
            elif atom.HasProp('_GasteigerCharge'):
                partial_charge = float(atom.GetProp("_GasteigerCharge"))
            else:
                assert False
            self.c[idx] = partial_charge

    def get(self, idx):
        assert self.c is not None
        return self.c[idx]


class EuclideanDistancePolicy:
    """
    Policy to handle distances in euclidean space.
    """
    def __init__(self):
        self.d = None
        self.distance_size = 0.3
        self.min_distance = 0.0
        self.max_distance = 10.2
        self.nof_distance_bins = int((self.max_distance - self.min_distance) / self.distance_size + 1)

    def discretize(self, dist):
        assert dist <= self.max_distance
        return int((dist - self.min_distance) / self.distance_size)

    def invalid(self):
        return self.nof_distance_bins

    def set_molecule(self, m):
        n = m.GetNumAtoms()
        self.d = np.zeros((n, n))
        self.d.fill(np.inf)
        conformer = m.GetConformer()
        for i in range(n):
            pos1 = conformer.GetAtomPosition(i)
            for j in range(n):
                pos2 = conformer.GetAtomPosition(j)
                self.d[i, j] = pos1.Distance(pos2)

    def get(self, idx1, idx2):
        assert self.d is not None
        return self.d[idx1, idx2]


class TopologicalDistancePolicy:
    """
    Policy to handle topological distances on the molecule graph.
    """
    def __init__(self, k):
        self.d = None
        self.nof_distance_bins = k

    def set_molecule(self, m):
        n = m.GetNumAtoms()
        self.d = np.zeros((n, n))
        self.d.fill(np.inf)

        for i in range(n):
            visited = np.zeros(n)
            q = queue.Queue()
            q.put(i)
            self.d[i, i] = 0
            visited[i] = 1

            while not q.empty():
                curr_id = q.get()
                curr_atom = m.GetAtomWithIdx(curr_id)
                for next_atom in curr_atom.GetNeighbors():
                    next_id = next_atom.GetIdx()
                    if not visited[next_id]:
                        visited[next_id] = 1
                        self.d[i, next_id] = self.d[i, curr_id] + 1
                        q.put(next_id)

    def discretize(self, dist):
        return dist

    def invalid(self):
        return self.nof_distance_bins

    def get(self, idx1, idx2):
        assert self.d is not None
        return self.d[idx1, idx2]


def parse_file_name(filename):
    """
    Parse the file name of a DUD mol2 file to get the target name and the y label
    :param filename: the filename string
    :return: protein target name, y_label string (ligand or decoy)
    """
    bname = os.path.basename(filename)
    splitted_bname = bname.split('_')

    if len(splitted_bname) == 3:
        target_name = splitted_bname[0]
        y_label_str = splitted_bname[1]
    elif len(splitted_bname) == 4:
        target_name = '_'.join([splitted_bname[0], splitted_bname[1]])
        y_label_str = splitted_bname[2]
    else:
        raise ValueError('File name has not expected format. Can not parse file name.')

    if y_label_str == 'decoys':
        y_label = 0
    elif y_label_str == 'ligands':
        y_label = 1
    else:
        raise ValueError('File name has not expected format. Can not parse file name.')

    return target_name, y_label


def read_mol2_file(mol2_file):
    """
    Reads molecules contained in a mol2 file
    :param mol2_file: the path to the mol2 file
    :return: list of molecule and number of molecules that could not be read
    """
    not_read = 0
    entry = []
    molecules = []

    with open(mol2_file, 'r') as f:

        for line in f:

            if line.startswith('@<TRIPOS>MOLECULE') and entry:
                mol = rdkit.Chem.rdmolfiles.MolFromMol2Block(''.join(entry), sanitize=True, removeHs=True)
                if mol:
                    molecules.append(mol)
                else:
                    not_read += 1
                entry = []
            entry.append(line)

        if entry:
            mol = rdkit.Chem.rdmolfiles.MolFromMol2Block(''.join(entry), sanitize=True, removeHs=True)
            if mol:
                molecules.append(mol)
            else:
                not_read += 1

    return molecules, not_read


def calculate_descriptor(molecule, type_policy, distance_policy, charge_policy, k=6):
    """
    Calculates the descriptor for a molecule on the basis of type, distance and charge policies.
    :param molecule: a molecule
    :param type_policy: a policy to handle atom types
    :param distance_policy: a policy to handle distances
    :param charge_policy: a policy to handle charges
    :param k: the number of local atoms to consider for each atom in the molecule (size of atom context)
    :return: the descriptor as list of tuples containing the features for each atom
    """
    type_policy.set_molecule(molecule)
    distance_policy.set_molecule(molecule)
    charge_policy.set_molecule(molecule)
    assert molecule.GetNumAtoms() >= k

    desc = []
    for idx in range(molecule.GetNumAtoms()):
        atom_type = type_policy.get(idx)
        partial_charge = charge_policy.get(idx)
        desc.append((idx, atom_type, partial_charge))

    result = []

    for idx1, atom_type1, partial_charge1 in desc:
        neighbours = []

        for idx2, atom_type2, partial_charge2 in desc:
            dist = distance_policy.get(idx1, idx2)
            neighbours.append((dist, atom_type2, partial_charge2))

        neighbours.sort(key=operator.itemgetter(0))
        curr_distances = [e[0] for e in neighbours[:k]]
        curr_atom_types = [e[1] for e in neighbours[:k]]
        curr_partial_charges = [e[2] for e in neighbours[:k]]

        for atom_type in curr_atom_types:
            if atom_type not in TypePolicy.all_atom_types:
                TypePolicy.all_atom_types[atom_type] = len(TypePolicy.all_atom_types)

        result.append((curr_atom_types, curr_distances, curr_partial_charges))

    return result


def refactor_descriptor(descriptor, type_policy, distance_policy, charge_policy):
    """
    Refactors/discretizes the provided descriptor according to the given policies.
    :param descriptor: molecular descriptor
    :param type_policy: a atom type policy
    :param distance_policy: a distance policy
    :param charge_policy: a charge policy
    :return: the refactored descriptor with feature values discretized
    """
    new_descriptor = []
    for curr_atom_types, curr_distances, curr_partial_charges in descriptor:
        new_atom_types = [type_policy.discretize(a) for a in curr_atom_types]
        new_distances = [distance_policy.discretize(d) for d in curr_distances]
        new_partial_charges = [charge_policy.discretize(c) for c in curr_partial_charges]
        new_descriptor.append((new_atom_types, new_distances, new_partial_charges))

    return new_descriptor


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--input', nargs='+', type=str, required=True,
                        help='input files to process')
    parser.add_argument('--reset', action='store_true',
                        help='Reset properties stored in molecule file')
    parser.add_argument('--output', type=str, required=True,
                        help='output file (.csv)')
    parser.add_argument('--topological', action='store_true',
                        help='use topological instead of euclidean distance')
    parser.add_argument('-k', '--k', type=int, default=6,
                        help='Number of neighboring atoms to consider')

    args = parser.parse_args()

    all_descriptors = []

    max_num_atoms = 0

    for filename in args.input:

        code, y_label = parse_file_name(filename)

        molecules, not_read = read_mol2_file(filename)
        print('Molecules read {}, not read {} from {}'.format(len(molecules), not_read, filename))

        descriptors = []

        for m in molecules:
            if not m:
                print('skipping invalid molecule')
                continue

            if args.reset:
                smiles = rdkit.Chem.MolToSmiles(m, isomericSmiles=True)
                m = None
                m1 = rdkit.Chem.MolFromSmiles(smiles)
                if not m1:
                    print('skipping: failed to read molecule from smiles!')
                    continue

                m2 = rdkit.Chem.AddHs(m1)
                if not m2:
                    print('skipping: failed to add hydrogens!')
                    continue

                re = rdkit.Chem.AllChem.EmbedMolecule(m2, rdkit.Chem.AllChem.ETKDG())
                if not m2.GetNumConformers() or re:
                    print('skipping: could not generate conformer!')
                    continue
                try:
                    rdkit.Chem.rdPartialCharges.ComputeGasteigerCharges(m2, throwOnParamFailure=True)
                except ValueError as e:
                    print('skipping: error on calculating Gasteiger charges, {}'.format(e))
                    continue
                m = m2

            charge_policy = ChargePolicy()
            type_policy = TypePolicy()

            if not args.topological:
                distance_policy = EuclideanDistancePolicy()
            else:
                distance_policy = TopologicalDistancePolicy(args.k)

            d = calculate_descriptor(m, type_policy, distance_policy, charge_policy, args.k)
            descriptors.append(d)
            max_num_atoms = max(max_num_atoms, len(d))
        all_descriptors.append((code, y_label, descriptors))

    columns = []

    columns += ['type_{}_{}'.format(i, j) for i in range(max_num_atoms + 1) for j in range(args.k)]
    columns += ['dist_{}_{}'.format(i, j) for i in range(max_num_atoms + 1) for j in range(args.k)]
    columns += ['chrg_{}_{}'.format(i, j) for i in range(max_num_atoms + 1) for j in range(args.k)]
    columns += ['active', 'inactive']
    columns += ['code']

    df = pd.DataFrame(columns=columns)

    for code, y_label, descriptors in all_descriptors:
        rows = []
        for d in descriptors:

            type_policy = TypePolicy()
            charge_policy = ChargePolicy()

            if not args.topological:
                distance_policy = EuclideanDistancePolicy()
            else:
                distance_policy = TopologicalDistancePolicy(args.k)

            rd = refactor_descriptor(d, type_policy, distance_policy, charge_policy)
            row = []
            for r in rd:
                row += r[0]
            for i in range(len(rd), max_num_atoms + 1):
                row += [type_policy.invalid() for j in range(args.k)]

            for r in rd:
                row += r[1]
            for i in range(len(rd), max_num_atoms + 1):
                row += [distance_policy.invalid() for j in range(args.k)]

            for r in rd:
                row += r[2]
            for i in range(len(rd), max_num_atoms + 1):
                row += [charge_policy.invalid() for j in range(args.k)]

            row += [1 if y_label else 0, 1 if not y_label else 0]
            row += [code]
            rows.append(row)
        df2 = pd.DataFrame(data=rows, columns=columns)
        df = df.append(df2)

    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
