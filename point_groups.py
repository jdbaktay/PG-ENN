import numpy as np
from numpy.linalg import matrix_power
import itertools
import math

#Pack of generator sets (each within same rep) for different reps of same group
class Group_Rep_Pack(object):
    def __init__(self, generator_sets=[], name = None, group_order=None):
        if name != None:
            self.name = name
        else:
            print(f'I have no name ...')

        if group_order == None:
            self.group_order = 0
            self.determine_order_auto = True

        else:
            self.determine_order_auto = True
            print(f'Determining group order automatically from largest Gen_Set ...')

        self.generator_dict = {}
        self.order_dict = {}

        for generator_set in generator_sets:
            self.add_generator_set(generator_set)
        
    def add_generator_set(self, gs):
        self.generator_dict[gs.name] = gs.generators
        self.order_dict[gs.name] = gs.orders
        num_unique = gs.num_unique

        if self.determine_order_auto:
            if self.group_order == None:
                self.group_order = num_unique
            elif num_unique > self.group_order:
                self.group_order = num_unique
        else:
            if num_unique > self.group_order:
                print(f'Num of unique elements of irrep {gs.name} > group order {self.group_order}!')

            if num_unique < self.group_order:
                print(f'Note: {gs.name} is unfaithful')
                ###How to incorporate unfaithful reps for dirls formula?###

    def return_element(self, irrep_name, gen_key):
        loc_gen = self.generator_dict[irrep_name]
        assert len(gen_key)==len(loc_gen), f'{gen_key} is not the same length as {loc_gen}'
        x = np.eye(loc_gen[0].shape[0])
        for s,n in zip(loc_gen, gen_key):
            x = np.matmul(x, matrix_power(s, int(n))) #We multiply from left to right
        return x
        
#Set of generators for same rep
class Generator_Set(object):
    def __init__(self, generators = [], name = None, dims = None):
        if name != None:
            self.name = name
        else:
            print(f'I have no name ...')

        if dims != None:
            print('No dims specified, auto-detecting')

        self.dims = dims
        self.generators = []
        self.orders = []

        for generator in generators:
            self.add_generator(generator)

    
    def add_generator(self, generator, order = None):
        if self.dims == None:
            self.dims = generator.shape
        else:
            assert generator.shape == self.dims, f'Incompatible generator {generator} for specified Gen_Set dim {self.dims}'

        if order == None:
            for n in range(2,128):
                print(matrix_power(generator,n))
                if np.allclose(matrix_power(generator,n),generator):
                    order = n-1
                    print(f'Computed order={order}!')
                    break
                if n==127: 
                    print('Order > 127, is it infinite?')
        self.generators.append(generator)
        self.orders.append(order)
    
    #Only generates unique elements, need to consider repeats for unfaithful reps
    @property
    def num_unique(self):
        return math.prod(self.orders)

class Point_Group(object):
    def __init__(self, point_group_label=None, generators=None):
        self.point_group_label = point_group_label
        self.order = None
        self.irrep_dims = None
        self.irrep_dict = None
        if generators != None:
            self.generate_props(generators)
        elif self.point_group_label != None:
            self.recall_props(self.point_group_label)
        else:
            print('No label or generators = No group props!')

    def generate_props(self, gen_pack):
        print('Generators not implemented yet')
        pass

    def irrep_labels(self):
        return list(self.irrep_dict.keys())
    
    def omega(self, n):
        return np.exp(2*np.pi*1.0j/n)

    def recall_props(self, label):
        if label in set(['Oh', 'O_h', 'Full Octahedral']):
            self.order = 48
            self.generate_props({})  
        if label in set(['D3', 'D_3', 'Dihedral 3']):
            self.order=6
            omega = self.omega(3)
            self.irrep_dims = {'E':2,'A1':1,'A2':1}
            self.irrep_dict = {'E': [ np.array([[1.0+0.0j,0],[0,1]]),
                                    np.array([[omega**2,0],[0,omega]]),
                                    np.array([[omega,0],[0,omega**2]]),
                                    np.array([[0,-1.0+0.0j],[-1,0]]),
                                    np.array([[0,-omega],[-omega**2,0]]),
                                    np.array([[0,-omega**2],[-omega,0]]),
                                    ],
                            'A1': [np.array([[1+0.0j]]),
                                    np.array([[1+0.0j]]),
                                    np.array([[1+0.0j]]),
                                    np.array([[1+0.0j]]),
                                    np.array([[1+0.0j]]),
                                    np.array([[1+0.0j]]),
                                    ],
                            'A2': [np.array([[1+0.0j]]),
                                    np.array([[1+0.0j]]),
                                    np.array([[1+0.0j]]),
                                    np.array([[-1+0.0j]]),
                                    np.array([[-1+0.0j]]),
                                    np.array([[-1+0.0j]]),                                 
                                    ]
                            }
            self.basis_dict = {'E': [np.array([[1],[0]]),np.array([[0],[1]])],
                            'A1':[np.array([1])],
                            'A2':[np.array([1])]}
            self.element_order = ['E','C3', 'C3^2','i','iC3', 'iC3^2']
        else:
            print(f"{label} point group label unrecognized, can't recall  properties!")

    def irrep(self, label):
        return self.irrep_dict[label]
    
    def irrep_element(self, irrep, element):
        if type(element) == str & element in set(self.element_order):
            pass
        elif type(element) == int:
            self.irrep(irrep)[element]  

    def coupling_coefficient(self, alpha, i, beta, j, gamma, n, i0=None,j0=None,n0=None):
        #a X b => C
        arep = self.irrep_dict[alpha]
        brep = self.irrep_dict[beta]
        crep = self.irrep_dict[gamma]
        da = self.irrep_dims[alpha]
        db = self.irrep_dims[beta]
        dc = self.irrep_dims[gamma]
        
        aindx = list(range(da))
        bindx = list(range(db))
        cindx = list(range(dc))
        abcindex = [aindx,bindx,cindx]

        if i0 == None:
            assert j0 == None and n0 == None
            print('Determining norm automagically...')
            for i0, j0, n0 in list(itertools.product(*abcindex)):
                norm_runner = 0.0+0.0j
                for g in range(6):
                    a = arep[g]
                    b = brep[g]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
                    c = crep[g]

                    norm_runner += a[i0][i0]*b[j0][j0]*np.conjugate(c[n0][n0])

                norm = ((dc/self.order)*norm_runner)**(1/2)
                if norm > 0.001:
                    print(f'Norm set to: {norm} with i0:{i0}, j0:{j0}, n0:{n0}')
                    break
        else:
            norm_runner = 0.0+0.0j
            for g in range(6):
                a = arep[g]
                b = brep[g]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
                c = crep[g]

                norm_runner += a[i0][i0]*b[j0][j0]*np.conjugate(c[n0][n0])

            norm = ((dc/self.order)*norm_runner)**(1/2)
            
        if norm < 0.001:
            print(f'Low norm: {norm}\nTrivial CG Likely!')

        runner = 0.0+0.0j
        for g in range(6):
            a = arep[g]
            b = brep[g]
            c = crep[g]

            runner += a[i][i0]*b[j][j0]*np.conjugate(c[n][n0])
            
        caibjgn =(dc/(self.order*norm))*runner
        return caibjgn