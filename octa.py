import numpy as np

from group_rep import GroupRepresentation
from dihedral import DihedralGroupD4

class OctahedralGroup(GroupRepresentation):
    def __init__(self):
        i = 1j
        eta = np.exp(2 * np.pi * i / 3)

        group_elements = [
            'I', 'C2x', 'C2y', 'C2z',
            'C+31', 'C+32', 'C+33', 'C+34',
            'C-31', 'C-32', 'C-33', 'C-34',
            'C+4x', 'C+4y', 'C+4z', 'C-4x',
            'C-4y', 'C-4z',
            'C`2a', 'C`2b', 'C`2c',
            'C`2d', 'C`2e', 'C`2f' ]

        character_table = {
            'A1': [1, 1, 1, 1, 1],
            'A2': [1, 1, 1, -1, -1],
            'E': [2, 2, -1, 0, 0],
            'T1': [3, -1, 0, 1, -1],
            'T2': [3, -1, 0, -1, 1]
        }

        all_irreps = {
            'A1': {
                zip(group_elements,
                    [np.array([[x]]) for x in np.ones(24, dtype=int)])
            },

            'A2': {
                zip(group_elements,
                [np.array([[x]]) for x in np.concatenate([np.ones(12, dtype=int),
                                                     -1 * np.ones(12, dtype=int)])])
            },

            'E': {
                'I': np.eye(2),
                'C2x': np.eye(2),
                'C2y': np.eye(2),
                'C2z': np.eye(2),
                'C+31': np.diagflat([eta, eta.conj()]),
                'C+32': np.diagflat([eta, eta.conj()]),
                'C+33': np.diagflat([eta, eta.conj()]),
                'C+34': np.diagflat([eta, eta.conj()]),
                'C-31': np.diagflat([eta.conj(), eta]),
                'C-32': np.diagflat([eta.conj(), eta]),
                'C-33': np.diagflat([eta.conj(), eta]),
                'C-34': np.diagflat([eta.conj(), eta]),
                'C+4x': np.array([[0, eta.conj()], [eta, 0]]),
                'C+4y': np.array([[0, eta], [eta.conj(), 0]]),
                'C+4z': np.array([[0, 1], [1, 0]]),
                'C-4x': np.array([[0, eta.conj()], [eta, 0]]),
                'C-4y': np.array([[0, eta], [eta.conj(), 0]]),
                'C-4z': np.array([[0, 1], [1, 0]]),
                'C`2a': np.array([[0, 1], [1, 0]]),
                'C`2b': np.array([[0, 1], [1, 0]]),
                'C`2c': np.array([[0, eta], [eta.conj(), 0]]),
                'C`2d': np.array([[0, eta.conj()], [eta, 0]]),
                'C`2e': np.array([[0, eta], [eta.conj(), 0]]),
                'C`2f': np.array([[0, eta.conj()], [eta, 0]])
            },
            'T1': {
                'I': np.eye(3),
                'C2x': np.diagflat([-1, -1, 1]),
                'C2y': np.diagflat([1, -1, -1]),
                'C2z': np.diagflat([-1, 1, -1]),
                'C+31': np.array([[0, 0, -i], [-i, 0, 0], [0, -1, 0]]),
                'C+32': np.array([[0, 0, -i], [i, 0, 0], [0, 1, 0]]),
                'C+33': np.array([[0, 0, i], [-i, 0, 0], [0, 1, 0]]),
                'C+34': np.array([[0, 0, i], [i, 0, 0], [0, -1, 0]]),
                'C-31': np.array([[0, i, 0], [0, 0, -1], [i, 0, 0]]),
                'C-32': np.array([[0, -i, 0], [0, 0, 1], [i, 0, 0]]),
                'C-33': np.array([[0, i, 0], [0, 0, 1], [-i, 0, 0]]),
                'C-34': np.array([[0, -i, 0], [0, 0, -1], [-i, 0, 0]]),
                'C+4x': np.array([[0, -i, 0], [-i, 0, 0], [0, 0, 1]]),
                'C+4y': np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),
                'C+4z': np.array([[0, 0, -i], [0, 1, 0], [-i, 0, 0]]),
                'C-4x': np.array([[0, i, 0], [i, 0, 0], [0, 0, 1]]),
                'C-4y': np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
                'C-4z': np.array([[0, 0, i], [0, 1, 0], [i, 0, 0]]),
                'C`2a': np.array([[0, 0, -i], [0, -1, 0], [i, 0, 0]]),
                'C`2b': np.array([[0, 0, i], [0, -1, 0], [-i, 0, 0]]),
                'C`2c': np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]]),
                'C`2d': np.array([[0, i, 0], [-i, 0, 0], [0, 0, -1]]),
                'C`2e': np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]),
                'C`2f': np.array([[0, -i, 0], [i, 0, 0], [0, 0, -1]]),
            },

            'T2': {
                'I': np.eye(3),
                'C2x': np.diagflat([-1, -1, 1]),
                'C2y': np.diagflat([1, -1, -1]),
                'C2z': np.diagflat([-1, 1, -1]),
                'C+31': np.array([[0, 0, -i], [-i, 0, 0], [0, -1, 0]]),
                'C+32': np.array([[0, 0, -i], [i, 0, 0], [0, 1, 0]]),
                'C+33': np.array([[0, 0, i], [-i, 0, 0], [0, 1, 0]]),
                'C+34': np.array([[0, 0, i], [i, 0, 0], [0, -1, 0]]),
                'C-31': np.array([[0, i, 0], [0, 0, -1], [i, 0, 0]]),
                'C-32': np.array([[0, -i, 0], [0, 0, 1], [i, 0, 0]]),
                'C-33': np.array([[0, i, 0], [0, 0, 1], [-i, 0, 0]]),
                'C-34': np.array([[0, -i, 0], [0, 0, -1], [-i, 0, 0]]),
                'C+4x': np.array([[0, i, 0], [i, 0, 0], [0, 0, -1]]),
                'C+4y': np.array([[-1, 0, 0], [0, 0, -1], [0, 1, 0]]),
                'C+4z': np.array([[0, 0, i], [0, -1, 0], [i, 0, 0]]),
                'C-4x': np.array([[0, -i, 0], [-i, 0, 0], [0, 0, -1]]),
                'C-4y': np.array([[-1, 0, 0], [0, 0, 1], [0, -1, 0]]),
                'C-4z': np.array([[0, 0, -i], [0, -1, 0], [-i, 0, 0]]),
                'C`2a': np.array([[0, 0, i], [0, 1, 0], [-i, 0, 0]]),
                'C`2b': np.array([[0, 0, -i], [0, 1, 0], [i, 0, 0]]),
                'C`2c': np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]),
                'C`2d': np.array([[0, -i, 0], [i, 0, 0], [0, 0, 1]]),
                'C`2e': np.array([[1, 0, 0], [0, 0, -1], [0, -1, 0]]),
                'C`2f': np.array([[0, i, 0], [-i, 0, 0], [0, 0, 1]]),
            }
        }

        irrep_dims = {
            'A1': 1,
            'A2': 1,
            'E': 2,
            'T1': 3,
            'T2': 3
        }

        super().__init__(group_elements, all_irreps, irrep_dims, character_table)

if __name__ == '__main__':
    O_group = OctahedralGroup()
    D4_group = DihedralGroupD4()

    # cg = O_group.calc_clebsch_gordan('E', 'T1', 'T2')

    cg = D4_group.calc_clebsch_gordan('B1', 'E', 'E')


    print(cg)
