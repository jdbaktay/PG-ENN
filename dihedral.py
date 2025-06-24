import numpy as np

from group_rep import GroupRepresentation

class DihedralGroupD4(GroupRepresentation):
    def __init__(self):
        i = 1j
        group_elements = ['I', 'C+4', 'C-4', 'C2',
                          'C`21', 'C`22', 'C``21', 'C``22'
        ]

        character_table = {
            'A1': [1, 1, 1, 1, 1],
            'A2': [1, 1, 1, -1, -1],
            'B1': [1, -1, 1, 1, -1],
            'B2': [1, -1, 1, -1, 1],
            'E': [2, 0, -2, 0, 0]
        }

        all_irreps = {
            'A1': dict(
                zip(group_elements,
                    [np.array([[x]]) for x in [1, 1, 1, 1, 1, 1, 1, 1]])
            ),
            'A2': dict(
                zip(group_elements,
                    [np.array([[x]]) for x in [1, 1, 1, 1, -1, -1, -1, -1]])
            ),
            'B1': dict(
                zip(group_elements,
                    [np.array([[x]]) for x in [1, -1, -1, 1, 1, 1, -1, -1]])
            ),
            'B2': dict(
                zip(group_elements,
                    [np.array([[x]]) for x in [1, -1, -1, 1, -1, -1, 1, 1]])
            ),
            'E': {
                'I': np.eye(2),
                'C+4': np.diagflat([-i, i]),
                'C-4': np.diagflat([i, -i]),
                'C2': np.diagflat([-1, -1]),
                'C`21': np.array([[0, -1], [-1, 0]]),
                'C`22': np.array([[0, 1], [1, 0]]),
                'C``21': np.array([[0, i], [-i, 0]]),
                'C``22': np.array([[0, -i], [i, 0]]),
            }
        }

        irrep_dims = {
            'A1': 1,
            'A2': 1,
            'B1': 1,
            'B2': 1,
            'E': 2
        }

        super().__init__(group_elements, all_irreps, irrep_dims, character_table)
