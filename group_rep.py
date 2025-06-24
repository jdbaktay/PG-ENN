import numpy as np

from numpy.linalg import eig

class GroupRepresentation:
    def __init__(
        self,
        group_elements: list,
        all_irreps: dict,
        irrep_dims: dict,
        character_table: dict
    ):
        self.group_elements = group_elements
        self.all_irreps = all_irreps
        self.irrep_dims = irrep_dims
        self.character_table = character_table

    def get_irrep(
            self,
            irrep_label: str,
            group_element: str | None = None
    ):
        if group_element == None:
            return self.all_irreps[irrep_label]
        else:
            return self.all_irreps[irrep_label][group_element]

    def _tensor_rep(self, irrep1, irrep2, g) -> np.ndarray:
        return np.kron(self.all_irreps[irrep1][g], self.all_irreps[irrep2][g])

    def _projector(self, irrep1, irrep2, target_irrep) -> np.ndarray:
        dims = self.irrep_dims

        d = dims[target_irrep]
        order = len(self.group_elements)
        size = dims[irrep1] * dims[irrep2]

        P = np.zeros((size, size), dtype=np.complex128)
        for g in list(self.group_elements):
            chi = np.trace(self.all_irreps[target_irrep][g])
            P += np.conj(chi) * self._tensor_rep(irrep1, irrep2, g)
        return (d / order) * P

    def calc_clebsch_gordan(
            self,
            irrep1: str,
            irrep2: str,
            target_irrep: str,
            tol: float = 1e-8
    ) -> np.ndarray:
        P = self._projector(irrep1, irrep2, target_irrep)
        vals, vecs = eig(P)
        CGs = []
        for i, val in enumerate(vals):
            if np.abs(val - 1) < tol:
                CGs.append(vecs[:, i] / np.linalg.norm(vecs[:, i]))
        return np.array(CGs)

    @property
    def list_irreps(self):
        return self.all_irreps.keys()

    @property
    def get_char_table(self):
        return self.character_table
