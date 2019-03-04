import numpy as np
import argparse
import rdkit
import rdkit.Chem
import rdkit.Chem.AllChem
import rdkit.Geometry
import os
from sklearn.externals import joblib


class Grid3D:

    def __init__(self, xmin, ymin, zmin, xmax, ymax, zmax, resolution):
        """
        Represents a uniformly spaced 3D grid of certain size
        :param xmin: minimal point of x axis
        :param ymin: minimal point of y axis
        :param zmin: minimal point of z axis
        :param xmax: maximal point of x axis
        :param ymax: maximal point of y axis
        :param zmax: maximal point of z axis
        :param resolution: the grids resolutions/spacing
        :param type: the data type represented by one grid point
        """
        assert(xmin < xmax and ymin < ymax and zmin < zmax and resolution > 0)
        self.xmin = np.float(xmin)
        self.ymin = np.float(ymin)
        self.zmin = np.float(zmin)
        self.xmax = np.float(xmax)
        self.ymax = np.float(ymax)
        self.zmax = np.float(zmax)
        self.resolution = np.float(resolution)

        self.nx = int((abs(self.xmax - self.xmin) / self.resolution)) + 1
        self.ny = int((abs(self.ymax - self.ymin) / self.resolution)) + 1
        self.nz = int((abs(self.zmax - self.zmin) / self.resolution)) + 1

    def contains_index(self, indices):
        if indices[0] < 0 or indices[0] >= self.nx:
            return False
        if indices[1] < 0 or indices[1] >= self.ny:
            return False
        if indices[2] < 0 or indices[2] >= self.nz:
            return False
        return True

    def get_bounding_box(self, point, radius):
        """
        Calculates the bounding box of a sphere relational to the grid
        :param point: sphere center
        :param radius: sphere radius
        :return: bounds per axis
        """
        x_upper = np.floor(((point[0] - self.xmin) + radius) / self.resolution)
        x_lower = np.ceil(((point[0] - self.xmin) - radius) / self.resolution)
        y_upper = np.floor(((point[1] - self.ymin) + radius) / self.resolution)
        y_lower = np.ceil(((point[1] - self.ymin) - radius) / self.resolution)
        z_upper = np.floor(((point[2] - self.zmin) + radius) / self.resolution)
        z_lower = np.ceil(((point[2] - self.zmin) - radius) / self.resolution)

        return x_upper, x_lower, y_upper, y_lower, z_upper, z_lower

    def get_grid_points_in_radius(self, point, radius):
        """
        Calculates all grid points contained in a sphere
        :param point: sphere center
        :param radius: sphere radius
        :return: list of tuples containing grid point coordinates x,y,z
        """
        x_upper, x_lower, y_upper, y_lower, z_upper, z_lower = self.get_bounding_box(point, radius)
        p = np.array(point)
        return [(x, y, z)
                for x in np.arange(x_lower, x_upper + 1)
                for y in np.arange(y_lower, y_upper + 1)
                for z in np.arange(z_lower, z_upper + 1)
                if np.linalg.norm(p - [self.xmin + (x * self.resolution),
                                       self.ymin + (y * self.resolution),
                                       self.zmin + (z * self.resolution)]) <= radius
                ]


def parse_file_name(filename):
    """
    Parses the filename of DUD-E files for protein target code and class label
    :param filename: name of file (needs to contain dir with target name)
    :return: target code and label
    """
    code = os.path.split(os.path.split(filename)[0])[1]

    bname = os.path.basename(filename)
    y_label_str = bname.split('_')[0]

    if y_label_str == 'decoys':
        y_label = 0
    elif y_label_str == 'actives':
        y_label = 1
    else:
        assert False

    return code, y_label


def main():
    parser = argparse.ArgumentParser(description='Calculates 3D-grid descriptor for a given molecule file.'
                                                 'The descriptor is stored in a sparse format in a binary output file.')

    parser.add_argument('--input', nargs='+', type=str, required=True,
                        help='input files to process')
    parser.add_argument('--output', type=str, required=True,
                        help='output binary file.')
    parser.add_argument('--use2D', action='store_true',
                        help='Use 2D conformation instead of a 3D one')

    args = parser.parse_args()

    if not args.use2D: # 3D
        resolution, xmin, ymin, zmin, xmax, ymax, zmax = 0.5, -11.5, -11.5, -11.5, 12, 12, 12
    else:  # 2D
        resolution, xmin, ymin, zmin, xmax, ymax, zmax = 0.5, -11.5, -11.5, 0, 12, 12, 0.01

    considered_elements = ['Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S']  # correspond to grid channels
    elem_indice_map = {e: i for i, e in enumerate(considered_elements)}

    all_descriptors = []

    grid = Grid3D(xmin=xmin, ymin=ymin, zmin=zmin, xmax=xmax, ymax=ymax, zmax=zmax,
                  resolution=resolution)

    for filename in args.input:

        code, y_label = parse_file_name(filename)
        print('Working on molecules of target {} with label {}'.format(code, y_label))

        descriptors = []

        for i, m in enumerate(rdkit.Chem.SmilesMolSupplier(filename, nameColumn=1, titleLine=0)):

            if not m:
                print('skipping invalid molecule')
                continue

            name = m.GetProp('_Name')

            if args.use2D:
                rdkit.Chem.AllChem.Compute2DCoords(m)
            else:
                m = rdkit.Chem.AddHs(m)
                if not m:
                    print('skipping: failed to add hydrogens!')
                    continue

                re = rdkit.Chem.AllChem.EmbedMolecule(m, rdkit.Chem.AllChem.ETKDG())
                if not m.GetNumConformers() or re:
                    print('skipping: could not generate conformer for {}!'.format(name))
                    continue

                # move mols centroid to the origin
                center = rdkit.Chem.rdMolTransforms.ComputeCentroid(m.GetConformers()[0])
                rdkit.Chem.rdMolTransforms.CanonicalizeConformer(m.GetConformers()[0], center=center, ignoreHs=True)

            conf = m.GetConformers()[0]  # we work with a single conformer only

            mol_descriptors = []

            for atom_id in range(conf.GetNumAtoms()):
                atom = m.GetAtomWithIdx(atom_id)

                a_sym = atom.GetSymbol()
                if a_sym not in elem_indice_map.keys():
                    # if atom not in the list, skip it
                    continue

                atom_pt = conf.GetAtomPosition(atom_id)
                if not args.use2D:
                    radius = rdkit.Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum())
                else:
                    radius = 1.5  # use constant atom radius

                for point in grid.get_grid_points_in_radius(atom_pt, radius):
                    if grid.contains_index(point):
                        if args.use2D:
                            point = point[:2]  # z is always zero
                        mol_descriptors.append(point + (elem_indice_map[a_sym],))

            mol_descriptors = list(set(mol_descriptors))  # remove duplicates
            mol_descriptors = np.array(mol_descriptors, dtype=np.int8)

            descriptors.append(mol_descriptors)

        all_descriptors.append((descriptors, code, y_label))

    grid_shape = (grid.nx, grid.ny, grid.nz, len(elem_indice_map))

    if args.use2D:
        grid_shape = (grid.nx, grid.ny, len(elem_indice_map))

    joblib.dump((all_descriptors, grid_shape), args.output)


if __name__ == "__main__":
    main()

