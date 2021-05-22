#!/usr/bin/env python
# coding: utf-8

# Moodle XML Question Generator, (C) 2019 Kosaku Nagasaka (Kobe University)

# ## モジュールファイルの生成

# In[13]:


if __name__ == "__main__":
    import subprocess
    subprocess.run(['jupyter', 'nbconvert', '--to', 'python', '_linear_algebra.ipynb','--output','linear_algebra.py'])


# # Helpers for Linear Algebra

# In[1]:


import math
import random
import sympy
import copy
if __name__ == "__main__":
    import IPython


# ## NOTES

# - Any matrix will be translated in the latex format with "pmatrix". Please translate it with your preferable environment (e.g. bmatrix) by the option of the top level function.
# - Any vector symbol will be translated in the latex format with "\vec". Please translate it with your preferable command (e.g. \boldsymbol) by the option of the top level function.

# ## minor helpers

# In[2]:


def partition_list0(vec, n):
    for i in range(0, len(vec), n):
        yield vec[i:i + n]
def partition_list(vec, n):
    return list(partition_list0(vec,n))


# In[ ]:


if __name__ == "__main__":
    print(partition_list([1,2,3,4,5,6,7,8,9,10],3))


# In[3]:


def thread_list0(vec, n):
    res = []
    for i in range(0, len(vec)):
        res += vec[i][n]
    return res
def thread_list(vec):
    return [thread_list0(vec, i) for i in range(0, len(vec[0]))]


# In[ ]:


if __name__ == "__main__":
    print(thread_list([[[1,2],[3,4]],[[5,6],[7,8]],[[10,11,12],[13,14,15]]]))


# In[4]:


def nonzero_randint(imin, imax):
    if 0 < imin or imax < 0:
        return random.randint(imin, imax)
    else:
        return random.choice(list(set(range(imin,imax+1)).difference(set([0]))))


# In[ ]:


if __name__ == "__main__":
    print([nonzero_randint(-15,-5) for i in range(20)])


# In[5]:


def rationalized_vector(vec):
    _denom = [sympy.denom(ele) for ele in vec]
    if len(_denom) > 1:
        _dlcm = sympy.ilcm(*_denom)
    else:
        _dlcm = _denom[0]
    return [_dlcm*ele for ele in vec]


# In[ ]:


if __name__ == "__main__":
    print(rationalized_vector([sympy.Rational(-3,7),-2,sympy.Rational(3,10)]))
    print(rationalized_vector([sympy.Rational(-3,7)]))


# In[6]:


def is_zero_vector(vec):
    for ele in vec:
        if ele != 0:
            return False
    return True


# In[ ]:


if __name__ == "__main__":
    print(is_zero_vector([1,2,3]), is_zero_vector([0,0,0]))


# In[7]:


def is_zero_vectors(vecs):
    for vec in vecs:
        if not is_zero_vector(vec):
            return False
    return True


# In[ ]:


if __name__ == "__main__":
    print(is_zero_vectors([[0,0,0],[1,2,3]]), is_zero_vectors([[0,0],[0,0,0]]))


# In[8]:


def is_integer_vector(vec):
    for ele in vec:
        if not sympy.sympify(ele).is_integer:
            return False
    return True


# In[ ]:


if __name__ == "__main__":
    print(is_integer_vector([1,2,3]), is_integer_vector([1,sympy.Rational(2,3),3]))


# In[9]:


def integer_to_quasi_square(n):
    fcts = sympy.Integer(n).factors()
    prms = []
    for p, e in fcts.items():
        for i in range(e):
            prms.append(p)
    r = 1
    for i in range(1,len(prms),2):
        r *= prms[i]
    c = 1
    for i in range(0,len(prms),2):
        c *= prms[i]
    return [r,c]


# In[ ]:


if __name__ == "__main__":
    print(integer_to_quasi_square(8))
    print(integer_to_quasi_square(7))


# In[10]:


def flatten_list(alist):
    rlist = []
    for lis in alist:
        rlist = rlist + lis
    return rlist


# In[ ]:


if __name__ == "__main__":
    print(flatten_list([[[1,2,3],[4,5]],[1,2]]))


# ## General Integer Set Operations

# In[ ]:


def eval_integer_set_expr(set_list):
    """Evaluation of the given expression of sets of integers in the original recursive structure."""
    if len(set_list) == 0:
        return {}
    elif set_list[0] == 'set':
        return set_list[1]
    elif set_list[0] == 'intersection':
        return eval_integer_set_expr(set_list[1]).intersection(eval_integer_set_expr(set_list[2]))
    elif set_list[0] == 'difference':
        return eval_integer_set_expr(set_list[1]).difference(eval_integer_set_expr(set_list[2]))
    elif set_list[0] == 'union':
        return eval_integer_set_expr(set_list[1]).union(eval_integer_set_expr(set_list[2]))


# In[ ]:


def generate_integer_set_expr(depth, universe, emin=1, emax=5):
    """Generate an expression of sets of integers in the original recursive structure."""
    if depth <= 0:
        return ['set',set(random.sample(universe,min(random.randint(emin,emax),len(universe))))]
    else:
        return [random.choice(['intersection','difference','union']), generate_integer_set_expr(depth-1, universe, emin, emax), generate_integer_set_expr(depth-1, universe, emin, emax)]       


# In[ ]:


def _integer_set_expr_to_text(set_list, is_top=False):
    prefix = '\\left(' if not(is_top) else ''
    suffix = '\\right)' if not(is_top) else ''
    if len(set_list) == 0:
        return sympy.latex({})
    elif set_list[0] == 'set':
        return sympy.latex(set_list[1])
    elif set_list[0] == 'intersection':
        return prefix + _integer_set_expr_to_text(set_list[1]) + '\\bigcap' + _integer_set_expr_to_text(set_list[2]) + suffix
    elif set_list[0] == 'difference':
        return prefix + _integer_set_expr_to_text(set_list[1]) + '\\setminus' + _integer_set_expr_to_text(set_list[2]) + suffix
    elif set_list[0] == 'union':
        return prefix + _integer_set_expr_to_text(set_list[1]) + '\\bigcup' + _integer_set_expr_to_text(set_list[2]) + suffix
def integer_set_expr_to_text(set_list, prefix='', suffix=''):
    """This converts the given integer set expresion into latex format with the math mode delimiters."""
    return r'\( '+prefix+ _integer_set_expr_to_text(set_list, True) +suffix+' \)'


# In[ ]:


if __name__ == "__main__":
    example_set_expr = generate_integer_set_expr(2, range(10), 1, 5)
    display(example_set_expr)


# In[ ]:


if __name__ == "__main__":
    IPython.display.display(IPython.display.HTML(integer_set_expr_to_text(example_set_expr)))


# In[ ]:


if __name__ == "__main__":
    display(eval_integer_set_expr(example_set_expr))


# ## Linear Space

# In[ ]:


class LinearSpace():
    """
    This generates a general linear space satisfying the specified condition.
    The internal representation is row vectors as generators.
    """
    def __init__(self, other=None):
        if other is not None:
            self.dim_min = copy.deepcopy(other.dim_min)
            self.dim_max = copy.deepcopy(other.dim_max)
            self.elem_min = copy.deepcopy(other.elem_min)
            self.elem_max = copy.deepcopy(other.elem_max)
            self.rnkde_min = copy.deepcopy(other.rnkde_min)
            self.rnkde_max = copy.deepcopy(other.rnkde_max)
            self.redun_min = copy.deepcopy(other.redun_min)
            self.redun_max = copy.deepcopy(other.redun_max)            
            self.dimension = copy.deepcopy(other.dimension)            
            self.num_disj = copy.deepcopy(other.num_disj)
            self.num_gens = copy.deepcopy(other.num_gens)
            self.is_keep_integer = copy.deepcopy(other.is_keep_integer)
            self.is_pivot_one = copy.deepcopy(other.is_pivot_one)
            self.is_pivot_rtl = copy.deepcopy(other.is_pivot_rtl)
            self.pivot_positions = copy.deepcopy(other.pivot_positions)
            self.basis = copy.deepcopy(other.basis)
            self.generator = copy.deepcopy(other.generator)
            self.coef_vectors = copy.deepcopy(other.coef_vectors)
            self.special_solution = copy.deepcopy(other.special_solution)
            self.constant_vector = copy.deepcopy(other.constant_vector)
            self.id = copy.deepcopy(other.id)
        else:
            self.dim_min = 1
            self.dim_max = 4
            self.elem_min = -3
            self.elem_max = +3
            self.rnkde_min = 1
            self.rnkde_max = 2
            self.redun_min = 1
            self.redun_max = 2            
            self.dimension = None            
            self.num_disj = None
            self.num_gens = None           
            self.is_keep_integer = True 
            self.is_pivot_one = False
            self.is_pivot_rtl = False
            self.pivot_positions = None
            self.basis = None
            self.generator = None
            self.coef_vectors = None
            self.special_solution = None
            self.constant_vector = None
            self.id = random.random()
    def __eq__(self, other):
        if not isinstance(other, LinearSpace):
            return False
        if self.dimension != other.dimension:
            return False
        if self.num_disj != other.num_disj:
            return False
        if sympy.Matrix(self.basis + other.basis).rank() != self.num_disj:
            return False
        return True
    def __hash__(self):
        self.id = hash(str(self.basis))
        return self.id
    def set_dimension_range(self, dmin, dmax):
        """
        The range of dimension of the whole vector space.
        """
        self.dim_min = dmin
        self.dim_max = dmax
    def set_element_range(self, emin, emax):
        """
        The range of elements of generators and other helper vectors.
        """
        self.elem_min = emin
        self.elem_max = emax
    def set_rank_deficient_range(self, rmin, rmax):
        """
        The range of deficient rank against to the dimension.
        """
        self.rnkde_min = rmin
        self.rnkde_max = rmax
    def set_redundant_range(self, rmin, rmax):
        """
        The range of number of redundant generators.
        """
        self.redun_min = rmin
        self.redun_max = rmax
    def generate_conditions(self):
        """
        This generates the fundamental conditions on the linear space randomly.
        """
        self.dimension = max(1,random.randint(self.dim_min, self.dim_max))
        self.num_disj = max(1,self.dimension - random.randint(self.rnkde_min, self.rnkde_max))
        self.num_gens = self.num_disj + random.randint(self.redun_min, self.redun_max)
    def set_keep_integer(self, true_or_false):
        """
        Any generated vector must be over integers if True.
        """
        self.is_keep_integer = true_or_false
    def set_pivot_one(self, true_or_false):
        """
        Any generated pivot on rref must be one if True.
        """
        self.is_pivot_one = true_or_false
    def set_pivot_rtl(self, true_or_false):
        """
        Any generated pivot on rref must be from right to left positions if True.
        """
        self.is_pivot_rtl = true_or_false
    def generate_basis(self):
        """
        This generates the basis vectors in echelon form.
        """
        if self.dimension is None:
            self.generate_conditions()
        self.pivot_positions = [0] + sorted(random.sample(range(1, self.dimension), self.num_disj - 1))
        self.basis = []        
        for p in self.pivot_positions:
            _echelon_form = [0 for i in range(self.dimension)]
            _echelon_form[p] = 1 if self.is_pivot_one else nonzero_randint(self.elem_min, self.elem_max)
            for i in range(p+1, self.dimension):
                if i not in self.pivot_positions:
                    _echelon_form[i] = nonzero_randint(self.elem_min, self.elem_max)
            self.basis.append(_echelon_form)
        if self.is_pivot_rtl:
            self.pivot_positions = [self.dimension - i - 1 for i in self.pivot_positions]
            self.basis = [list(reversed(v)) for v in reversed(self.basis)]
    def generate_generator(self):
        """
        This generates a generator which spannes the linear space of the basis.
        """
        if self.basis is None:
            self.generate_basis()
        self.generator = []
        for i in range(self.num_gens):
            _generator = [0 for j in range(self.dimension)]
            _cis = [nonzero_randint(self.elem_min, self.elem_max) for j in range(self.num_disj)]
            for idx in range(self.dimension):
                for ib in range(self.num_disj):
                    _generator[idx] += _cis[ib]*self.basis[ib][idx]
            self.generator.append(_generator)
        _matrix = sympy.Matrix(self.generator)
        if _matrix.rank() != self.num_disj:
            self.generate_generator()
    def generate_homogeneous_eq(self):
        """
        This generates a homogeneous linear equation whose solution space is this linear space.
        """
        if self.basis is None:
            self.generate_basis()
        _matrix = sympy.Matrix(self.basis)
        _cgenerator = [sympy.matrix2numpy(v.transpose()).tolist()[0] for v in _matrix.nullspace()]
        self.coef_vectors = []
        for i in range(self.dimension - self.num_disj + self.num_gens - self.num_disj):
            _coef = [0 for j in range(self.dimension)]
            _cis = [nonzero_randint(self.elem_min, self.elem_max) for j in range(self.dimension - self.num_disj)]
            for idx in range(self.dimension):
                for ib in range(self.dimension - self.num_disj):
                    _coef[idx] += _cis[ib]*_cgenerator[ib][idx]
            if self.is_keep_integer:
                self.coef_vectors.append(rationalized_vector(_coef))
            else:
                self.coef_vectors.append(_coef)
        _matrix = sympy.Matrix(self.coef_vectors)
        if _matrix.rank() != self.dimension - self.num_disj:
            self.generate_homogeneous_eq()
    def generate_nonhomogeneous_eq(self):
        """
        This generates a non-homogeneous linear equation whose basic solution space is this linear space.
        """
        if self.coef_vectors is None:
            self.generate_homogeneous_eq()
        self.special_solution = [nonzero_randint(self.elem_min, self.elem_max) for i in range(self.dimension)]
        self.constant_vector = [[0] for i in range(self.dimension - self.num_disj + self.num_gens - self.num_disj)]
        for i in range(self.dimension - self.num_disj + self.num_gens - self.num_disj):
            for j in range(self.dimension):
                self.constant_vector[i][0] += self.coef_vectors[i][j]*self.special_solution[j]       
    def generate(self):
        """
        This succesively generates several internal information on the linear space.
        """
        self.generate_conditions()
        self.generate_basis()
        self.generate_generator()
        self.generate_homogeneous_eq()
        self.generate_nonhomogeneous_eq()
    def a_spanned_vector(self, is_integer_coefs=None):
        """
        This generates a vector in this linear space (sub-space).
        """
        if self.generator is None:
            self.generate_generator()
        _spanned_vec = [0 for j in range(self.dimension)]
        if is_integer_coefs is None:
            _is_integer = self.is_keep_integer
        else:
            _is_integer = is_integer_coefs
        if _is_integer:
            _cis = [nonzero_randint(self.elem_min, self.elem_max) for j in range(self.num_disj)]
        else:            
            _cis = [sympy.Rational(nonzero_randint(self.elem_min, self.elem_max),
                                   nonzero_randint(self.elem_min, self.elem_max)) for j in range(self.num_disj)]
        for idx in range(self.dimension):
            for ib in range(self.num_disj):
                _spanned_vec[idx] += _cis[ib]*self.generator[ib][idx]
        return _spanned_vec
    def a_solution_vector(self, is_integer_coefs=None):
        """
        This generates a solution vector of the non-homogeneous linear equation (sols of homo-eq can be generated by a_spanned_vector).
        """
        if self.special_solution is None:
            self.generate_nonhomogeneous_eq()
        _a_basic_sol = self.a_spanned_vector(is_integer_coefs)
        return [self.special_solution[i] + _a_basic_sol[i] for i in range(self.dimension)]
    def is_a_spanned_vector(self, vec):
        """
        This returns True if the vector is included in the linear space.
        """
        if len(vec) > self.dimension:
            return False
        else:
            if len(vec) < self.dimension:
                _vec = vec + [0 for i in range(self.dimension - len(vec))]
                return sympy.Matrix(self.basis + [_vec]).rank() == self.num_disj
            else:
                return sympy.Matrix(self.basis + [vec]).rank() == self.num_disj
    def is_a_solution_vector(self, vec):
        """
        This returns True if the vector is included in the solution set.
        """
        if len(vec) != self.dimension:
            return False
        else:
            _substed = [[-self.constant_vector[i][0]] for i in range(self.dimension - self.num_disj + self.num_gens - self.num_disj)]
            for i in range(self.dimension - self.num_disj + self.num_gens - self.num_disj):
                for j in range(self.dimension):
                    _substed[i][0] += self.coef_vectors[i][j]*vec[j]
            for i in range(self.dimension - self.num_disj + self.num_gens - self.num_disj):
                if _substed[i][0] != 0:
                    return False
            return True
    def a_vector_as_str(self, vec, is_latex_closure=True, is_polynomial=False, is_matrix=False):
        """
        This generates a string expression of the given vector.
        """
        _x = sympy.symbols('x')
        if is_latex_closure:
            _text = r'\( '
        else:
            _text = ''
        veclen = len(vec)
        if is_matrix:
            _matrix_dim = integer_to_quasi_square(veclen)
        if is_matrix:
            _text += sympy.latex(sympy.Matrix(_matrix_dim[0], _matrix_dim[1], vec), mat_delim='', mat_str='pmatrix')
        elif is_polynomial:
            _text += sympy.latex(sum([vec[j]*_x**j for j in range(veclen)]))
        else:
            _text += sympy.latex(sympy.Matrix([vec]).transpose(), mat_delim='', mat_str='pmatrix')
        if is_latex_closure:
            _text += r' \)'
        return _text
    def str_as_spanned_space(self, domain='R', is_latex_closure=True, is_polynomial=False, is_matrix=False):
        """
        This generates a string expression of the linear space generated by the generators.
        """
        if is_matrix:
            _matrix_dim = integer_to_quasi_square(self.dimension)
        _x = sympy.symbols('x')
        if is_latex_closure:
            _text = r'\( '
        else:
            _text = ''
        _text += r'\left\{\left.'
        for i in range(self.num_gens):
            _text += r'c_{' + str(i+1) + r'}'
            if is_matrix:
                _text += sympy.latex(sympy.Matrix(_matrix_dim[0], _matrix_dim[1], self.generator[i]), mat_delim='', mat_str='pmatrix')
            elif is_polynomial:
                _text += r'(' + sympy.latex(sum([self.generator[i][j]*_x**j for j in range(self.dimension)])) + r')'
            else:
                _text += sympy.latex(sympy.Matrix([self.generator[i]]).transpose(), mat_delim='', mat_str='pmatrix')
            if i < self.num_gens - 1:
                _text += r'+'
        _text += r'\;\right|\;'
        for i in range(self.num_gens):
            _text += r'c_{' + str(i+1) + r'}'
            if i < self.num_gens - 1:
                _text += r','
        _text += r'\in\mathbb{' + domain + '}'
        _text += r'\right\}'
        if is_latex_closure:
            _text += r' \)'
        return _text        
    def str_as_solution_space(self, domain='R', is_latex_closure=True, is_homogeneous=True):
        """
        This generates a string expression of the linear space of the solution set.
        """
        if is_latex_closure:
            _text = r'\( '
        else:
            _text = ''
        _text += r'\left\{\vec{x}'
        _text += r'\in\mathbb{' + domain + '}'            
        _text += r'^{' + str(self.dimension) + r'}\;\left|\;'        
        _text += sympy.latex(sympy.Matrix(self.coef_vectors), mat_delim='', mat_str='pmatrix') + r'\vec{x} = '
        if is_homogeneous:
            _text += r'\vec{0}'
        else:
            _text += sympy.latex(sympy.Matrix(self.constant_vector), mat_delim='', mat_str='pmatrix') 
        _text += r'\right.\right\}'
        if is_latex_closure:
            _text += r' \)'        
        return _text
    def str_of_linear_mapping_as_image_space(self, domain='R', is_latex_closure=True):
        """
        This generates a string expression of the linear mapping as the image space.
        """
        if is_latex_closure:
            _text = r'\( '
        else:
            _text = ''
        _text += r'f:\mathbb{' + domain + '}^{' + str(self.num_gens) + r'}'    
        _text += r'\rightarrow\mathbb{'+ domain + '}^{' + str(self.dimension) + r'}'    
        _text += r',\;\vec{x}\mapsto '
        _text += sympy.latex(sympy.Matrix(self.generator).transpose(), mat_delim='', mat_str='pmatrix') + r'\vec{x}'
        if is_latex_closure:
            _text += r' \)'
        return _text        
    def str_of_linear_mapping_as_kernel_space(self, domain='R', is_latex_closure=True):
        """
        UNDER CONSTRUCTION
        This generates a string expression of the linear mapping as the kernel space.
        """
        if is_latex_closure:
            _text = r'\( '
        else:
            _text = ''
        _text += r'f:\mathbb{' + domain + '}^{' + str(self.dimension) + r'}'    
        _text += r'\rightarrow\mathbb{'+ domain + '}^{' + str(len(self.coef_vectors)) + r'}'    
        _text += r',\;\vec{x}\mapsto '
        _text += sympy.latex(sympy.Matrix(self.coef_vectors), mat_delim='', mat_str='pmatrix') + r'\vec{x}'
        if is_latex_closure:
            _text += r' \)'
        return _text


# In[ ]:


if __name__ == "__main__":
    ls = LinearSpace()
    ls.set_dimension_range(1,4)
    ls.set_element_range(-3,3)
    ls.set_rank_deficient_range(1,2)
    ls.set_redundant_range(1,2)
    ls.set_keep_integer(False)
    ls.generate()
    print("dimension: ", ls.dimension)
    print("num_disj: ", ls.num_disj)
    print("num_gens: ", ls.num_gens)           
    print("is_keep_integer: ", ls.is_keep_integer) 
    print("pivot_positions: ", ls.pivot_positions) 
    print("basis: ", ls.basis) 
    print("generator: ", ls.generator) 
    print("coef_vectors: ", ls.coef_vectors) 
    print("special_solution: ", ls.special_solution) 
    print("constant_vector: ", ls.constant_vector) 
    print("sols: ",ls.a_solution_vector())
    print("span: ",ls.a_spanned_vector())
    IPython.display.display(IPython.display.HTML(ls.str_as_spanned_space()))
    IPython.display.display(IPython.display.HTML(ls.str_as_spanned_space(domain='Z')))
    IPython.display.display(IPython.display.HTML(ls.str_as_spanned_space(is_polynomial=True)))
    IPython.display.display(IPython.display.HTML(ls.str_as_spanned_space(is_matrix=True)))
    IPython.display.display(IPython.display.HTML(ls.str_as_solution_space()))
    IPython.display.display(IPython.display.HTML(ls.str_as_solution_space(domain='Z')))
    IPython.display.display(IPython.display.HTML(ls.str_as_solution_space(is_homogeneous=False)))
    IPython.display.display(IPython.display.HTML(ls.str_of_linear_mapping_as_image_space()))
    IPython.display.display(IPython.display.HTML(ls.str_of_linear_mapping_as_kernel_space()))


# In[ ]:


if __name__ == "__main__":
    print(ls.is_a_spanned_vector([sympy.Rational(1,2),sympy.Rational(3,2),sympy.Rational(1,2)]))


# In[ ]:


if __name__ == "__main__":
    print(ls.is_a_solution_vector([[7],[3]]))


# ## Orthogonal Complement in Rn

# In[646]:


class OrthogonalComplement():
    """
    This generates a general linear space as an orthogonal complement in Rn.
    The internal representation is row vectors as generators.
    """
    def __init__(self, other=None):
        if other is not None:
            self.dim_min = copy.deepcopy(other.dim_min)
            self.dim_max = copy.deepcopy(other.dim_max)
            self.elem_min = copy.deepcopy(other.elem_min)
            self.elem_max = copy.deepcopy(other.elem_max)
            self.rnkde_min = copy.deepcopy(other.rnkde_min)
            self.rnkde_max = copy.deepcopy(other.rnkde_max)
            self.redun_min = copy.deepcopy(other.redun_min)
            self.redun_max = copy.deepcopy(other.redun_max)            
            self.dimension = copy.deepcopy(other.dimension)            
            self.num_disj = copy.deepcopy(other.num_disj)
            self.num_gens = copy.deepcopy(other.num_gens)
            self.pivot_positions = copy.deepcopy(other.pivot_positions)
            self.is_for_solution_space = copy.deepcopy(other.is_for_solution_space)
            self.basis = copy.deepcopy(other.basis)
            self.given_space = copy.deepcopy(other.generator)
            self.id = copy.deepcopy(other.id)
        else:
            self.dim_min = 1
            self.dim_max = 4
            self.elem_min = -3
            self.elem_max = +3
            self.rnkde_min = 1
            self.rnkde_max = 2
            self.redun_min = 1
            self.redun_max = 2            
            self.dimension = None            
            self.num_disj = None
            self.num_gens = None
            self.pivot_positions = None
            self.is_for_solution_space = None
            self.basis = None
            self.given_space = None
            self.id = random.random()
    def __eq__(self, other):
        if not isinstance(other, OrthogonalComplement):
            return False
        if self.dimension != other.dimension:
            return False
        if self.num_disj != other.num_disj:
            return False
        if sympy.Matrix(self.basis + other.basis).rank() != self.num_disj:
            return False
        return True
    def __hash__(self):
        self.id = hash(str(self.basis) + str(self.given_space))
        return self.id
    def set_dimension_range(self, dmin, dmax):
        """
        The range of dimension of the whole vector space.
        """
        self.dim_min = dmin
        self.dim_max = dmax
    def set_element_range(self, emin, emax):
        """
        The range of elements of generators and other helper vectors.
        """
        self.elem_min = emin
        self.elem_max = emax
    def set_rank_deficient_range(self, rmin, rmax):
        """
        The range of deficient rank against to the dimension.
        """
        self.rnkde_min = rmin
        self.rnkde_max = rmax
    def set_redundant_range(self, rmin, rmax):
        """
        The range of number of redundant generators.
        """
        self.redun_min = rmin
        self.redun_max = rmax
    def generate_conditions(self):
        """
        This generates the fundamental conditions on the linear space randomly.
        """
        self.dimension = max(1,random.randint(self.dim_min, self.dim_max))
        self.num_disj = max(1,self.dimension - random.randint(self.rnkde_min, self.rnkde_max))
        self.is_for_solution_space = True if random.random() < 0.5 else False
        if self.is_for_solution_space:
            self.num_gens = self.num_disj + random.randint(self.redun_min, self.redun_max)
        else:
            self.num_gens = (self.dimension - self.num_disj) + random.randint(self.redun_min, self.redun_max)
    def generate_basis(self):
        """
        This generates the basis vectors in echelon form.
        """
        if self.dimension is None:
            self.generate_conditions()
        self.pivot_positions = [0] + sorted(random.sample(range(1, self.dimension), self.num_disj - 1))
        self.basis = []        
        for p in self.pivot_positions:
            _echelon_form = [0 for i in range(self.dimension)]
            _echelon_form[p] = 1
            for i in range(p+1, self.dimension):
                if i not in self.pivot_positions:
                    _echelon_form[i] = nonzero_randint(self.elem_min, self.elem_max)
            self.basis.append(_echelon_form)
        self.basis = [list(reversed(e)) for e in reversed(self.basis)]
    def generate_given_space(self):
        """
        This generates a given space for which the basis is of its orthogonal complement.
        """
        if self.basis is None:
            self.generate_basis()
        if self.is_for_solution_space:
            _basis = self.basis
        else:
            _pivots = [self.dimension - p - 1 for p in reversed(self.pivot_positions)]
            _fill_basis = []
            for i in range(self.dimension):
                if i in _pivots:
                    _fill_basis.append(self.basis[_pivots.index(i)])
                else:
                    _fill_basis.append([0 for j in range(self.dimension)])
            _mat = sympy.eye(self.dimension) - sympy.Matrix(_fill_basis).transpose()
            _basis = []
            for _list in sympy.matrix2numpy(_mat).tolist():
                if sum([abs(e) for e in _list]) != 0:
                    _basis.append(_list)
        self.given_space = []
        for i in range(self.num_gens):
            _generator = [0 for j in range(self.dimension)]
            _cis = [nonzero_randint(self.elem_min, self.elem_max) for j in range(len(_basis))]
            for idx in range(self.dimension):
                for ib in range(len(_basis)):
                    _generator[idx] += _cis[ib]*_basis[ib][idx]
            self.given_space.append(_generator)
        _matrix = sympy.Matrix(self.given_space)
        if _matrix.rank() != len(_basis):
            self.generate_given_space()
    def generate(self):
        """
        This succesively generates several internal information on the linear space.
        """
        self.generate_conditions()
        self.generate_basis()
        self.generate_given_space()
    def is_a_basis(self, vecs):
        """
        This returns True if the given vectors is one of alternative basis of the complement.
        """
        if len(vecs) != len(self.basis):
            return False
        elif len(vecs[0]) != len(self.basis[0]):
            return False
        elif sympy.Matrix(vecs).rank() != len(self.basis):
            return False
        elif sympy.Matrix(self.basis + vecs).rank() != len(self.basis):
            return False
        return True
    def str_a_vector(self, vec, is_latex_closure=True):
        """
        This generates a string expression of the given vector.
        """
        if is_latex_closure:
            _text = r'\( '
        else:
            _text = ''
        _text += sympy.latex(sympy.Matrix([vec]).transpose(), mat_delim='', mat_str='pmatrix')
        if is_latex_closure:
            _text += r' \)'
        return _text
    def str_given_space(self, is_latex_closure=True):
        if self.is_for_solution_space:
            return self.str_given_space_as_solution_space(is_latex_closure)
        else:
            return self.str_given_space_as_spanned_space(is_latex_closure)    
    def str_given_space_as_solution_space(self, is_latex_closure=True):
        """
        This generates a string expression of the linear space of the solution set.
        """
        if is_latex_closure:
            _text = r'\( '
        else:
            _text = ''
        _text += r'\left\{\begin{array}{c|c}\vec{x}\in\mathbb{R}^{' + str(self.dimension) + r'}&'        
        _text += sympy.latex(sympy.Matrix(self.given_space), mat_delim='', mat_str='pmatrix') + r'\vec{x} = \vec{0}\end{array}\right\}'
        if is_latex_closure:
            _text += r' \)'        
        return _text
    def str_given_space_as_spanned_space(self, is_latex_closure=True):
        """
        This generates a string expression of the linear space generated by the generators.
        """
        if is_latex_closure:
            _text = r'\( '
        else:
            _text = ''
        _text += r'\left\{\begin{array}{c|c}'
        for i in range(len(self.given_space)):
            _text += r'c_{' + str(i+1) + r'}'
            _text += sympy.latex(sympy.Matrix([self.given_space[i]]).transpose(), mat_delim='', mat_str='pmatrix')
            if i < len(self.given_space) - 1:
                _text += r'+'
        _text += r'&'
        for i in range(len(self.given_space)):
            _text += r'c_{' + str(i+1) + r'}'
            if i < len(self.given_space) - 1:
                _text += r','
        _text += r'\in\mathbb{R}\end{array}\right\}'
        if is_latex_closure:
            _text += r' \)'
        return _text    
    def str_orthogonal_complement(self, is_latex_closure=True):
        """
        This generates a string expression of the orthogonal complement.
        """
        if is_latex_closure:
            _text = r'\( '
        else:
            _text = ''
        _text += r'\left\{\begin{array}{c|c}'
        for i in range(len(self.basis)):
            _text += r'c_{' + str(i+1) + r'}'
            _text += sympy.latex(sympy.Matrix([self.basis[i]]).transpose(), mat_delim='', mat_str='pmatrix')
            if i < len(self.basis) - 1:
                _text += r'+'
        _text += r'&'
        for i in range(len(self.basis)):
            _text += r'c_{' + str(i+1) + r'}'
            if i < len(self.basis) - 1:
                _text += r','
        _text += r'\in\mathbb{R}\end{array}\right\}'
        if is_latex_closure:
            _text += r' \)'
        return _text    


# In[672]:


if __name__ == "__main__":
    oc = OrthogonalComplement()
    oc.set_dimension_range(2,4)
    oc.set_element_range(-3,3)
    oc.set_rank_deficient_range(1,2)
    oc.set_redundant_range(1,2)
    oc.generate()
    print("dimension: ", oc.dimension)
    print("num_disj: ", oc.num_disj)
    print("num_gens: ", oc.num_gens)
    print("basis: ", oc.basis) 
    print("is_for_solution_space: ", oc.is_for_solution_space)
    print("given_space: ", oc.given_space) 
    IPython.display.display(IPython.display.HTML(oc.str_given_space()))
    IPython.display.display(IPython.display.HTML(oc.str_orthogonal_complement()))


# ## sub space or Not sub space

# In[ ]:


class LinearSubSpace():
    """
    This is an instance for generating a subspace or a non-subspace in several representations.
    """
    all_subspace_type = ['generator', 'equation', 'polynomial', 'matrix', 'constraint']
    def __init__(self):
        self.is_subspace = None
        self.dim_min = 1
        self.dim_max = 3
        self.elem_min = -5
        self.elem_max = +5
        self.dimension = None
        self.subspace_type = None
        self.data = None   
        self.id = random.random()
    def __eq__(self, other):
        if not isinstance(other, LinearSubSpace):
            return False
        elif self.dimension != other.dimension:
            return False
        elif self.subspace_type != other.subspace_type:
            return False
        elif self.data != other.data:
            return False
        return True
    def __hash__(self):
        self.id = hash(str(self.dimension) + str(self.subspace_type) + str(self.data))
        return self.id
    def set_dimension_range(self, dmin, dmax):
        """
        The range of dimension of the whole vector space or length.
        """
        self.dim_min = dmin
        self.dim_max = dmax
    def set_element_range(self, emin, emax):
        """
        The range of elements of generators and other helper vectors.
        """
        self.elem_min = emin
        self.elem_max = emax
    def generate(self, is_subspace=True, subspace_type=None):
        """
        This generates the internal data for the subspace or similar set.
        """
        self.is_subspace = is_subspace
        if subspace_type is None:
            self.subspace_type = random.choice(LinearSubSpace.all_subspace_type)
        else:
            self.subspace_type = subspace_type
        if self.subspace_type == 'generator':
            self.generate_generator()
        elif self.subspace_type == 'equation':
            self.generate_equation()
        elif self.subspace_type == 'polynomial':
            self.generate_polynomial()
        elif self.subspace_type == 'matrix':
            self.generate_matrix()
        else: # 'constraint'
            self.generate_constraint()
    def str(self, is_latex_closure=True):
        """
        This generates a latex expression of the subspace or similar set.
        """
        if self.subspace_type == 'generator':
            return self.str_generator(is_latex_closure)
        elif self.subspace_type == 'equation':
            return self.str_equation(is_latex_closure)
        elif self.subspace_type == 'polynomial':
            return self.str_polynomial(is_latex_closure)
        elif self.subspace_type == 'matrix':
            return self.str_matrix(is_latex_closure)
        else: # 'constraint'
            return self.str_constraint(is_latex_closure)
    def generate_generator(self):
        self.dimension = [random.randint(self.dim_min, self.dim_max) for i in range(2)]
        if not self.is_subspace:
            if self.dimension[0] < 2:
                self.dimension[0] = 2
        self.data = {}
        if self.is_subspace:
            self.data['nvar'] = self.dimension[0]
        else:
            self.data['nvar'] = self.dimension[0] - 1
        self.data['gens'] = [[random.randint(self.elem_min, self.elem_max) for i in range(self.dimension[1])] for j in range(self.dimension[0])]
        if not self.is_subspace:
            if sympy.Matrix(self.data['gens'][:-1]).rank() == sympy.Matrix(self.data['gens']).rank():
                self.generate_generator()
    def str_generator(self, is_latex_closure):
        if is_latex_closure:
            _text = r'\( '
        else:
            _text = ''
        _text += r'\left\{\left.'
        for i in range(self.dimension[0]):
            if i < self.data['nvar']:
                _text += r'c_{' + str(i+1) + r'}'
            _text += sympy.latex(sympy.Matrix([self.data['gens'][i]]).transpose(), mat_delim='', mat_str='pmatrix')
            if i < self.dimension[0] - 1:
                _text += r'+'
        _text += r'\;\right|\;'
        for i in range(self.dimension[0]):
            if i < self.data['nvar']:
                _text += r'c_{' + str(i+1) + r'}'
            if i < self.data['nvar'] - 1:
                _text += r','
        _text += r'\in\mathbb{R}'
        _text += r'\right\}'
        if is_latex_closure:
            _text += r' \)'
        return _text
    def generate_equation(self):
        self.dimension = [random.randint(self.dim_min, self.dim_max) for i in range(2)]
        self.data = {}
        if self.is_subspace:
            self.data['const'] = [[0] for i in range(self.dimension[0])]
        else:
            self.data['const'] = [[nonzero_randint(self.elem_min, self.elem_max)] for i in range(self.dimension[0])]
        self.data['gens'] = [[random.randint(self.elem_min, self.elem_max) for i in range(self.dimension[1])] for j in range(self.dimension[0])]
    def str_equation(self, is_latex_closure):
        if is_latex_closure:
            _text = r'\( '
        else:
            _text = ''
        _text += r'\left\{\vec{x}\in\mathbb{R}'            
        _text += r'^{' + str(self.dimension[1]) + r'}\;\left|\;'
        _text += sympy.latex(sympy.Matrix(self.data['gens']), mat_delim='', mat_str='pmatrix') + r'\vec{x} = ' + sympy.latex(sympy.Matrix(self.data['const']), mat_delim='', mat_str='pmatrix') 
        _text += r'\right.\right\}'
        if is_latex_closure:
            _text += r' \)'        
        return _text
    def generate_polynomial(self):
        self.dimension = [random.randint(self.dim_min, self.dim_max) for i in range(2)]
        if not self.is_subspace:
            if self.dimension[0] < 2:
                self.dimension[0] = 2
        self.data = {}
        if self.is_subspace:
            self.data['nvar'] = self.dimension[0]
        else:
            self.data['nvar'] = self.dimension[0] - 1
        self.data['gens'] = [[random.randint(self.elem_min, self.elem_max) for i in range(self.dimension[1])] for j in range(self.dimension[0])]
        if not self.is_subspace:
            if sympy.Matrix(self.data['gens'][:-1]).rank() == sympy.Matrix(self.data['gens']).rank():
                self.generate_generator()        
    def str_polynomial(self, is_latex_closure):
        _x = sympy.symbols('x')
        if is_latex_closure:
            _text = r'\( '
        else:
            _text = ''
        _text += r'\left\{\left.'
        for i in range(self.dimension[0]):
            if i < self.data['nvar']:
                _text += r'c_{' + str(i+1) + r'}'
            _text += r'(' + sympy.latex(sum([self.data['gens'][i][j]*_x**j for j in range(self.dimension[1])])) + r')'
            if i < self.dimension[0] - 1:
                _text += r'+'
        _text += r'\;\right|\;'
        for i in range(self.dimension[0]):
            if i < self.data['nvar']:
                _text += r'c_{' + str(i+1) + r'}'
            if i < self.data['nvar'] - 1:
                _text += r','
        _text += r'\in\mathbb{R}'
        _text += r'\right\}'
        if is_latex_closure:
            _text += r' \)'
        return _text
    def generate_matrix(self):
        self.dimension = [random.randint(self.dim_min, self.dim_max) for i in range(3)]
        if not self.is_subspace:
            if self.dimension[0] < 2:
                self.dimension[0] = 2
        self.data = {}
        if self.is_subspace:
            self.data['nvar'] = self.dimension[0]
        else:
            self.data['nvar'] = self.dimension[0] - 1
        self.data['gens'] = [[[random.randint(self.elem_min, self.elem_max) for i in range(self.dimension[2])] for j in range(self.dimension[1])] for k in range(self.dimension[0])]
        if not self.is_subspace:
            _matrix = []
            for k in range(self.dimension[0]):
                _vec = []
                for j in range(self.dimension[1]):
                    _vec += self.data['gens'][k][j]
                _matrix.append(_vec.copy())
            if sympy.Matrix(_matrix[:-1]).rank() == sympy.Matrix(_matrix).rank():
                self.generate_matrix()        
    def str_matrix(self, is_latex_closure):
        if is_latex_closure:
            _text = r'\( '
        else:
            _text = ''
        _text += r'\left\{\left.'
        for i in range(self.dimension[0]):
            if i < self.data['nvar']:
                _text += r'c_{' + str(i+1) + r'}'
            _text += sympy.latex(sympy.Matrix(self.data['gens'][i]), mat_delim='', mat_str='pmatrix')
            if i < self.dimension[0] - 1:
                _text += r'+'
        _text += r'\;\right|\;'
        for i in range(self.dimension[0]):
            if i < self.data['nvar']:
                _text += r'c_{' + str(i+1) + r'}'
            if i < self.data['nvar'] - 1:
                _text += r','
        _text += r'\in\mathbb{R}'
        _text += r'\right\}'
        if is_latex_closure:
            _text += r' \)'
        return _text
    def generate_constraint(self):
        self.dimension = [random.randint(self.dim_min, self.dim_max) for i in range(2)]
        if self.dimension[1] < 2:
            self.dimension[1] = 2
        self.data = {}
        if self.is_subspace:
            self.data['const'] = [0 for i in range(self.dimension[0])]
        else:
            self.data['const'] = [nonzero_randint(self.elem_min, self.elem_max) for i in range(self.dimension[0])]
        self.data['gens'] = random.sample(range(-self.dimension[0],self.dimension[0]+1), self.dimension[0])
    def str_constraint(self, is_latex_closure):
        if is_latex_closure:
            _text = r'\( '
        else:
            _text = ''
        _text += r'\left\{f(x)\in\mathbb{R}[x]'            
        _text += r'_{' + str(self.dimension[1]-1) + r'}\;\left|\;'
        for i in range(self.dimension[0]):
            _text += r'f(' + str(self.data['gens'][i]) + r')=' + str(self.data['const'][i])
            if i < self.dimension[0] - 1:
                _text += r','        
        _text += r'\right.\right\}'
        if is_latex_closure:
            _text += r' \)'        
        return _text


# In[ ]:


if __name__ == "__main__":
    for i in range(10):
        lss = LinearSubSpace()
        lss.generate(is_subspace=True)
        print(lss.subspace_type)
        IPython.display.display(IPython.display.HTML(lss.str()))


# ## Linear Map or Not Linear Map

# In[ ]:


class LinearMap():
    """
    This is an instance for generating a linear map or a non-linear map in several representations.
    """
    domain_type = ['numeric', 'matrix', 'polynomial']
    representation_type = ['symbolic', 'element-wise']
    map_type = ['linear', 'non-linear-polynomial', 'non-zero-constant']
    def __init__(self, other=None):
        if other is not None:
            self.source_domain = copy.deepcopy(other.source_domain)
            self.destination_domain = copy.deepcopy(other.destination_domain)
            self.representation = copy.deepcopy(other.representation)
            self.map = copy.deepcopy(other.map)
            self.source_dim = copy.deepcopy(other.source_dim)
            self.destination_dim = copy.deepcopy(other.destination_dim)
            self.dim_min = copy.deepcopy(other.dim_min)
            self.dim_max = copy.deepcopy(other.dim_max)
            self.iddim_min = copy.deepcopy(other.iddim_min)
            self.iddim_max = copy.deepcopy(other.iddim_max)
            self.elem_min = copy.deepcopy(other.elem_min)
            self.elem_max = copy.deepcopy(other.elem_max)
            self.image_dim = copy.deepcopy(other.image_dim)
            self.kernel_dim = copy.deepcopy(other.kernel_dim)
            self.data4nl = copy.deepcopy(other.data4nl)
            self.rref = copy.deepcopy(other.rref)
            self.matrix = copy.deepcopy(other.matrix)
            self.representation_matrix = copy.deepcopy(other.representation_matrix)
            self.source_basis = copy.deepcopy(other.source_basis)
            self.destination_basis = copy.deepcopy(other.destination_basis)
            self.pivot_positions = copy.deepcopy(other.pivot_positions)
            self.is_force_same_space = copy.deepcopy(other.is_force_same_space)
            self.id = copy.deepcopy(other.id)
        else:
            self.source_domain = self.domain_type[0]
            self.destination_domain = self.domain_type[0]
            self.representation = self.representation_type[0]
            self.map = self.map_type[0]
            self.source_dim = 2
            self.destination_dim = 2
            self.dim_min = 2
            self.dim_max = 2
            self.iddim_min = 0
            self.iddim_max = 2
            self.elem_min = -5
            self.elem_max = 5
            self.image_dim = 0
            self.kernel_dim = 0
            self.data4nl = None
            self.rref = None
            self.matrix = None
            self.representation_matrix = None
            self.source_basis = None
            self.destination_basis = None
            self.pivot_positions = None
            self.is_force_same_space = False
            self.id = 0
    def __eq__(self, other):
        if not isinstance(other, LinearMap):
            return False
        elif self.source_domain != other.source_domain or self.destination_domain != other.destination_domain:
            return False
        elif self.source_dim != other.source_dim or self.destination_dim != other.destination_dim:
            return False
        elif self.dim_min != other.dim_min or self.dim_max != other.dim_max:
            return False
        elif self.iddim_min != other.iddim_min or self.iddim_max != other.iddim_max:
            return False
        elif self.elem_min != other.elem_min or self.elem_max != other.elem_max:
            return False
        elif self.image_dim != other.image_dim or self.kernel_dim != other.kernel_dim or self.data4nl != other.data4nl:
            return False
        elif self.rref != other.rref or self.matrix != other.matrix:
            return False
        elif self.representation_matrix != other.representation_matrix or self.is_force_same_space != other.is_force_same_space:
            return False
        elif self.source_basis != other.source_basis or self.destination_basis != other.destination_basis:
            return False
        return True
    def __hash__(self):
        self.id = hash(self.source_domain + self.destination_domain 
                       + str(self.source_dim) + str(self.destination_dim) 
                       + str(self.dim_min) + str(self.dim_max) 
                       + str(self.iddim_min) + str(self.iddim_max)
                       + str(self.elem_min) + str(self.elem_max)
                       + str(self.image_dim) + str(self.kernel_dim)
                       + str(self.data4nl) + str(self.rref) + str(self.matrix)
                       + str(self.representation_matrix) + str(self.is_force_same_space)
                       + str(self.source_basis) + str(self.destination_basis))
        return self.id
    def set_dimension_range(self, dmin, dmax):
        """
        The range of dimension of the ground vector space or length.
        """
        self.dim_min = dmin
        self.dim_max = dmax
    def set_image_dimension_deficient_range(self, iddmin, iddmax):
        """
        The range of image dimension deficient of the map.
        """
        self.iddim_min = iddmin
        self.iddim_max = iddmax
    def set_element_range(self, emin, emax):
        """
        The range of elements of generators and other helper vectors.
        """
        self.elem_min = emin
        self.elem_max = emax
    def generate_config(self, is_linear=True, is_force_symbolic=False, is_force_numeric=False, is_force_same_space=False):
        self.is_force_same_space = is_force_same_space
        self.source_dim = random.randint(self.dim_min, self.dim_max)
        if is_force_same_space:
            self.destination_dim = self.source_dim
        else:
            self.destination_dim = random.randint(self.dim_min, self.dim_max)
        self.image_dim = max(0, min(self.source_dim, self.destination_dim) - random.randint(self.iddim_min, self.iddim_max))
        self.kernel_dim = self.source_dim - self.image_dim
        if is_linear:
            self.map = 'linear'
        elif random.random() <= 0.5:
            self.map = 'non-linear-polynomial'
        else:
            self.map = 'non-zero-constant'
        if is_force_symbolic:
            self.representation = 'symbolic'
        elif self.map == 'non-linear-polynomial':
            self.representation = 'element-wise'
        else:
            self.representation = random.choice(self.representation_type)
        if is_force_numeric:
            self.source_domain = 'numeric'
            self.destination_domain = 'numeric'
        elif self.representation == 'symbolic':
            self.source_domain = self.domain_type[random.randint(0,1)]
            while self.map == 'linear' and self.source_domain == 'matrix':
                self.source_domain = self.domain_type[random.randint(0,1)]
            self.destination_domain = self.source_domain
        elif random.random() < 0.5:
            self.source_domain = random.choice(self.domain_type)
            while integer_to_quasi_square(self.source_dim)[0] == 1 and self.source_domain == 'matrix':
                self.source_domain = random.choice(self.domain_type)
            self.destination_domain = self.source_domain
        else:
            self.source_domain = random.choice(self.domain_type)
            while integer_to_quasi_square(self.source_dim)[0] == 1 and self.source_domain == 'matrix':
                self.source_domain = random.choice(self.domain_type)
            self.destination_domain = random.choice(self.domain_type)
        if is_force_same_space:
            self.destination_domain = self.source_domain
    def generate_rref(self):
        """
        This generates the reduced row echelon form of the matrix rep.
        """
        if self.image_dim == 0:
            self.pivot_positions = []
        else:
            self.pivot_positions = [0] + sorted(random.sample(range(1, self.source_dim), self.image_dim - 1))
        self.rref = []
        for p in self.pivot_positions:
            _echelon_form = [0 for i in range(self.source_dim)]
            _echelon_form[p] = 1
            for i in range(p+1, self.source_dim):
                if i not in self.pivot_positions:
                    _echelon_form[i] = nonzero_randint(self.elem_min, self.elem_max)
            self.rref.append(_echelon_form)
    def generate_matrix(self):
        """
        This generates a matrix representation.
        """
        if self.rref is None:
            self.generate_rref()
        self.matrix = []
        if self.rref == []:
            self.matrix = [[0 for j in range(self.source_dim)] for i in range(self.destination_dim)]
        else:
            for i in range(self.destination_dim):
                _generator = [0 for j in range(self.source_dim)]
                _cis = [nonzero_randint(self.elem_min, self.elem_max) for j in range(self.image_dim)]
                for idx in range(self.source_dim):
                    for ib in range(self.image_dim):
                        _generator[idx] += _cis[ib]*self.rref[ib][idx]
                self.matrix.append(_generator)
            _matrix = sympy.Matrix(self.matrix)
            if _matrix.rank() != self.image_dim:
                self.generate_matrix()
    def generate_data_for_non_linear(self):
        """
        This generates a data for non-linear map.
        [degK matrix, ..., deg1 matrix, deg0 matrix] for non-linear-polynomial.
        non zero vector of destination_dim for non-zero-constant.
        """
        self.data4nl = []
        if self.map == 'non-linear-polynomial':
            if random.random() < 0.5: # deg=1
                _data4nl_deg = [[random.randint(self.elem_min, self.elem_max) for j in range(self.source_dim)] for i in range(self.destination_dim)]
                self.data4nl.append(_data4nl_deg)
                _data4nl_deg = [[random.randint(self.elem_min, self.elem_max) for j in range(self.source_dim)] for i in range(self.destination_dim)]
                while sum(sub_list.count(0) for sub_list in _data4nl_deg) == self.source_dim*self.destination_dim:
                    _data4nl_deg = [[random.randint(self.elem_min, self.elem_max) for j in range(self.source_dim)] for i in range(self.destination_dim)]
                self.data4nl.append(_data4nl_deg)
            else: # deg=K
                _degree = random.randint(2,3)
                _data4nl_deg = [[random.randint(self.elem_min, self.elem_max) for j in range(self.source_dim)] for i in range(self.destination_dim)]
                while sum(sub_list.count(0) for sub_list in _data4nl_deg) == self.source_dim*self.destination_dim:
                    _data4nl_deg = [[random.randint(self.elem_min, self.elem_max) for j in range(self.source_dim)] for i in range(self.destination_dim)]
                self.data4nl.append(_data4nl_deg)
                for i in range(_degree - 1):
                    _data4nl_deg = [[random.randint(self.elem_min, self.elem_max) for j in range(self.source_dim)] for i in range(self.destination_dim)]
                    self.data4nl.append(_data4nl_deg)
                _data4nl_deg = [[0 for j in range(self.source_dim)] for i in range(self.destination_dim)]
                self.data4nl.append(_data4nl_deg)
        else: #self.map == 'non-zero-constant'
            self.data4nl = [random.randint(self.elem_min, self.elem_max) for i in range(self.destination_dim)]
            while self.data4nl.count(0) == self.destination_dim:
                self.data4nl = [random.randint(self.elem_min, self.elem_max) for i in range(self.destination_dim)]
    def generate_basis(self, is_source_standard_basis=True, is_destination_standard_basis=True):
        if is_source_standard_basis:
            self.source_basis = [[1 if i == j else 0 for i in range(self.source_dim)] for j in range(self.source_dim)]
        else:
            _matrix = sympy.Matrix([])
            while _matrix.rank() != self.source_dim:
                self.source_basis = [[random.randint(self.elem_min, self.elem_max) for j in range(self.source_dim)] for i in range(self.source_dim)]
                _matrix = sympy.Matrix(self.source_basis)            
        if is_destination_standard_basis:
            self.destination_basis = [[1 if i == j else 0 for i in range(self.destination_dim)] for j in range(self.destination_dim)]
        else:
            _matrix = sympy.Matrix([])
            while _matrix.rank() != self.destination_dim or abs(_matrix.det()) != 1:
                self.destination_basis = [[1 if i >= j else 0 for i in range(self.destination_dim)] for j in range(self.destination_dim)]
                for i in range(self.destination_dim):
                    _cis = [random.randint(-1,1) for j in range(self.destination_dim)]
                    for idx_row in range(self.destination_dim):
                        if i == idx_row:
                            continue
                        for idx_col in range(self.destination_dim):
                            self.destination_basis[i][idx_col] += _cis[idx_row]*self.destination_basis[idx_row][idx_col]
                _matrix = sympy.Matrix(self.destination_basis)
        if self.is_force_same_space:
            self.source_basis = self.destination_basis
    def generate_representation_matrix(self):
        self.representation_matrix = sympy.matrix2numpy(sympy.Matrix(self.destination_basis).transpose().inv()*sympy.Matrix(self.matrix)*sympy.Matrix(self.source_basis).transpose()).tolist()
    def generate(self, is_linear=True, is_source_standard_basis=True, is_destination_standard_basis=True, is_force_symbolic=False, is_force_numeric=False, is_force_same_space=False):
        """
        This generates the internal data for the map.
        """
        if self.source_dim != self.image_dim + self.kernel_dim:
            self.generate_config(is_linear, is_force_symbolic, is_force_numeric, is_force_same_space)
        if is_linear:
            self.generate_rref()
            self.generate_matrix()
            self.generate_basis(is_source_standard_basis, is_destination_standard_basis)
            self.generate_representation_matrix()
        else:
            self.generate_data_for_non_linear()
    def _str_domain(self, dom, dim, is_latex_closure=False):
        if is_latex_closure:
            _text = r'\( '
        else:
            _text = ''
        if dom == 'numeric':
            _text += r'\mathbb{R}^{' + str(dim) + r'}'
        elif dom == 'matrix':
            sd = integer_to_quasi_square(dim) # sd = [r,c]
            _text += r'\mathbb{R}^{' + str(sd[0]) + r'\times' + str(sd[1]) + r'}'
        else: # 'polynomial'
            _text += r'\mathbb{R}[x]_{' + str(dim - 1) + r'}'
        if is_latex_closure:
            _text += r' \)'
        return _text
    def _str_source_element_num(self):
        if self.representation == 'symbolic':
            return r'\vec{x}'
        _text = r'\begin{pmatrix}'
        for i in range(self.source_dim):
            _text += r'x_{' + str(i+1) + r'} \\'    
        _text += r'\end{pmatrix}'
        return _text
    def _str_source_element_mat(self):
        if self.representation == 'symbolic':
            return r'M'
        _text = '' 
        sd = integer_to_quasi_square(self.source_dim) # sd = [r,c]
        _text = r'\begin{pmatrix}'
        for i in range(sd[0]):
            for j in range(sd[1]):
                _text += r'm_{' + str(i*sd[1]+j+1) + r'}'    
            if j < sd[1] - 1:
                _text += r'&'
            else:
                _text += r'\\'
        _text += r'\end{pmatrix}'
        return _text
    def _str_source_element_pol(self):
        _text = ''
        for i in range(self.source_dim - 1, -1, -1):
            if i < self.source_dim - 1:
                _text += r'+'
            if i > 1:
                _text += r'a_{' + str(i+1) + r'}x^{' + str(i) + r'}'    
            elif i > 0:
                _text += r'a_{' + str(i+1) + r'}x'   
            else:
                _text += r'a_{' + str(i+1) + r'}'   
        return _text    
    def str_source_element(self, is_latex_closure=False):
        if is_latex_closure:
            _text = r'\( '
        else:
            _text = ''
        if self.source_domain == 'numeric':
            _text += self._str_source_element_num()
        elif self.source_domain == 'matrix':
            _text += self._str_source_element_mat()
        else: # 'polynomial'       
            _text += self._str_source_element_pol()
        if is_latex_closure:
            _text += r' \)'
        return _text  
    def _str_destination_element_num(self):
        _text = ''
        if self.map == 'non-zero-constant':
            _text += sympy.latex(sympy.Matrix([self.data4nl]).transpose(), mat_delim='', mat_str='pmatrix')
        elif self.representation == 'symbolic': # and 'linear'
            _text += sympy.latex(sympy.Matrix(self.matrix), mat_delim='', mat_str='pmatrix') + r'\vec{x}'            
        elif self.map == 'linear': # and 'element-wise'
            _matrix = sympy.Matrix(self.matrix)
            _vars = []
            if self.source_domain == 'numeric':
                for i in range(self.source_dim):
                    _vars.append(sympy.Symbol('x'+str(i+1)))
            elif self.source_domain == 'matrix':
                sd = integer_to_quasi_square(self.source_dim) # sd = [r,c]
                for i in range(sd[0]):
                    for j in range(sd[1]):
                        _vars.append(sympy.Symbol('m'+str(i*sd[1]+j+1)))
            else: # 'polynomial'
                for i in range(self.source_dim):
                    _vars.append(sympy.Symbol('a'+str(i+1)))
            _vector = sympy.Matrix([[v] for v in _vars])
            _text += sympy.latex(_matrix*_vector, mat_delim='', mat_str='pmatrix')      
        else: # 'non-linear-polynomial' and 'element-wise'
            _vars = []
            if self.source_domain == 'numeric':
                for i in range(self.source_dim):
                    _vars.append(sympy.Symbol('x'+str(i+1)))
            elif self.source_domain == 'matrix':
                sd = integer_to_quasi_square(self.source_dim) # sd = [r,c]
                for i in range(sd[0]):
                    for j in range(sd[1]):
                        _vars.append(sympy.Symbol('m'+str(i*sd[1]+j+1)))
            else: # 'polynomial'
                for i in range(self.source_dim):
                    _vars.append(sympy.Symbol('a'+str(i+1)))
            _result = sympy.Matrix([[0] for i in range(self.destination_dim)])
            for k in range(len(self.data4nl) - 1, -1, -1):
                _matrix = sympy.Matrix(self.data4nl[len(self.data4nl) - 1 - k])
                _vector = sympy.Matrix([[v**k] for v in _vars])
                _result += _matrix*_vector
            _text += sympy.latex(_result, mat_delim='', mat_str='pmatrix')
        return _text
    def _str_destination_element_mat(self):
        _text = ''
        sd = integer_to_quasi_square(self.destination_dim) # sd = [r,c]
        if self.map == 'non-zero-constant':
            _text += sympy.latex(sympy.Matrix(partition_list(self.data4nl, sd[1])), mat_delim='', mat_str='pmatrix')
        elif self.map == 'linear': # and 'element-wise'
            _matrix = sympy.Matrix(self.matrix)
            _vars = []
            if self.source_domain == 'numeric':
                for i in range(self.source_dim):
                    _vars.append(sympy.Symbol('x'+str(i+1)))
            elif self.source_domain == 'matrix':
                source_sd = integer_to_quasi_square(self.source_dim) # sd = [r,c]
                for i in range(source_sd[0]):
                    for j in range(source_sd[1]):
                        _vars.append(sympy.Symbol('m'+str(i*source_sd[1]+j+1)))
            else: # 'polynomial'
                for i in range(self.source_dim):
                    _vars.append(sympy.Symbol('a'+str(i+1)))
            _vector = sympy.Matrix([[v] for v in _vars])
            _result = _matrix*_vector
            _text += sympy.latex(sympy.Matrix(partition_list([e for e in _result], sd[1])), mat_delim='', mat_str='pmatrix')      
        else: # 'non-linear-polynomial' and 'element-wise'
            _vars = []
            if self.source_domain == 'numeric':
                for i in range(self.source_dim):
                    _vars.append(sympy.Symbol('x'+str(i+1)))
            elif self.source_domain == 'matrix':
                source_sd = integer_to_quasi_square(self.source_dim) # sd = [r,c]
                for i in range(source_sd[0]):
                    for j in range(source_sd[1]):
                        _vars.append(sympy.Symbol('m'+str(i*source_sd[1]+j+1)))
            else: # 'polynomial'
                for i in range(self.source_dim):
                    _vars.append(sympy.Symbol('a'+str(i+1)))
            _result = sympy.Matrix([[0] for i in range(self.destination_dim)])
            for k in range(len(self.data4nl) - 1, -1, -1):
                _matrix = sympy.Matrix(self.data4nl[len(self.data4nl) - 1 - k])
                _vector = sympy.Matrix([[v**k] for v in _vars])
                _result += _matrix*_vector
            _text += sympy.latex(sympy.Matrix(partition_list([e for e in _result], sd[1])), mat_delim='', mat_str='pmatrix')    
        return _text
    def _str_destination_element_pol(self):
        _text = ''
        if self.map == 'non-zero-constant':
            _result = sympy.Matrix([self.data4nl]).transpose()
        elif self.map == 'linear': # and 'element-wise'
            _matrix = sympy.Matrix(self.matrix)
            _vars = []
            if self.source_domain == 'numeric':
                for i in range(self.source_dim):
                    _vars.append(sympy.Symbol('x'+str(i+1)))
            elif self.source_domain == 'matrix':
                sd = integer_to_quasi_square(self.source_dim) # sd = [r,c]
                for i in range(sd[0]):
                    for j in range(sd[1]):
                        _vars.append(sympy.Symbol('m'+str(i*sd[1]+j+1)))
            else: # 'polynomial'
                for i in range(self.source_dim):
                    _vars.append(sympy.Symbol('a'+str(i+1)))
            _vector = sympy.Matrix([[v] for v in _vars])
            _result = _matrix*_vector   
        else: # 'non-linear-polynomial' and 'element-wise'
            _vars = []
            if self.source_domain == 'numeric':
                for i in range(self.source_dim):
                    _vars.append(sympy.Symbol('x'+str(i+1)))
            elif self.source_domain == 'matrix':
                sd = integer_to_quasi_square(self.source_dim) # sd = [r,c]
                for i in range(sd[0]):
                    for j in range(sd[1]):
                        _vars.append(sympy.Symbol('m'+str(i*sd[1]+j+1)))
            else: # 'polynomial'
                for i in range(self.source_dim):
                    _vars.append(sympy.Symbol('a'+str(i+1)))
            _result = sympy.Matrix([[0] for i in range(self.destination_dim)])
            for k in range(len(self.data4nl) - 1, -1, -1):
                _matrix = sympy.Matrix(self.data4nl[len(self.data4nl) - 1 - k])
                _vector = sympy.Matrix([[v**k] for v in _vars])
                _result += _matrix*_vector
        _x = sympy.Symbol('x')
        _rmatrix = sympy.Matrix([[_x**i for i in range(0, self.destination_dim)]])*_result
        _text += sympy.latex(_rmatrix[0,0])
        return _text    
    def str_destination_element(self, is_latex_closure=False):
        if is_latex_closure:
            _text = r'\( '
        else:
            _text = ''
        if self.destination_domain == 'numeric':
            _text += self._str_destination_element_num()
        elif self.destination_domain == 'matrix':
            _text += self._str_destination_element_mat()
        else: # 'polynomial'       
            _text += self._str_destination_element_pol()
        if is_latex_closure:
            _text += r' \)'
        return _text
    def str_map(self, is_latex_closure=True):
        """
        This generates a latex expression of the map.
        """
        if is_latex_closure:
            _text = r'\( '
        else:
            _text = ''
        _text += r'f:' + self._str_domain(self.source_domain, self.source_dim)
        _text += r'\rightarrow ' + self._str_domain(self.destination_domain, self.destination_dim)
        _text += r',\;'
        _text += self.str_source_element()
        _text += r'\mapsto '
        _text += self.str_destination_element()
        if is_latex_closure:
            _text += r' \)'
        return _text
    def str_source_domain(self, is_latex_closure=True):
        return self._str_domain(self.source_domain, self.source_dim)
    def str_destination_domain(self, is_latex_closure=True):
        return self._str_domain(self.destination_domain, self.destination_dim)
    def _str_basis_num(self, basis, dim):
        _text = r'\left\{'
        for i in range(dim):
            _text += sympy.latex(sympy.Matrix([basis[i]]).transpose(), mat_delim='', mat_str='pmatrix')
            if i < dim - 1:
                _text += r',\;'
        _text += r'\right\}'
        return _text
    def _str_basis_mat(self, basis, dim):
        _text = r'\left\{'
        sd = integer_to_quasi_square(len(basis[0])) # sd = [r,c]
        for i in range(dim):
            _text += sympy.latex(sympy.Matrix(partition_list(basis[i],sd[1])), mat_delim='', mat_str='pmatrix')
            if i < dim - 1:
                _text += r',\;'
        _text += r'\right\}'
        return _text
    def _str_basis_pol(self, basis, dim):
        _text = r'\left\{'
        for i in range(dim):
            pol = 0
            _x = sympy.Symbol('x')
            for j in range(len(basis[0])):
                pol += basis[i][j]*_x**j
            _text += sympy.latex(pol)
            if i < dim - 1:
                _text += r',\;'        
        _text += r'\right\}'
        return _text
    def str_source_basis(self, is_latex_closure=True):
        if is_latex_closure:
            _text = r'\( '
        else:
            _text = ''
        if self.source_domain == 'numeric':
            _text += self._str_basis_num(self.source_basis, self.source_dim)
        elif self.source_domain == 'matrix':
            _text += self._str_basis_mat(self.source_basis, self.source_dim)
        else: # 'polynomial'       
            _text += self._str_basis_pol(self.source_basis, self.source_dim)
        if is_latex_closure:
            _text += r' \)'
        return _text
    def str_image_of_source_basis(self, is_latex_closure=True):
        _basis = sympy.Matrix(self.source_basis).transpose()
        _basis = sympy.Matrix(self.matrix)*_basis
        _basis = sympy.matrix2numpy(_basis.transpose()).tolist()
        if is_latex_closure:
            _text = r'\( '
        else:
            _text = ''
        if self.destination_domain == 'numeric':
            _text += self._str_basis_num(_basis, self.source_dim)
        elif self.destination_domain == 'matrix':
            _text += self._str_basis_mat(_basis, self.source_dim)
        else: # 'polynomial'       
            _text += self._str_basis_pol(_basis, self.source_dim)
        if is_latex_closure:
            _text += r' \)'
        return _text
    def str_image_of_standard_basis(self, is_latex_closure=True):
        _basis = sympy.eye(self.source_dim).transpose()
        _basis = sympy.Matrix(self.matrix)*_basis
        _basis = sympy.matrix2numpy(_basis.transpose()).tolist()
        if is_latex_closure:
            _text = r'\( '
        else:
            _text = ''
        if self.destination_domain == 'numeric':
            _text += self._str_basis_num(_basis, self.source_dim)
        elif self.destination_domain == 'matrix':
            _text += self._str_basis_mat(_basis, self.source_dim)
        else: # 'polynomial'       
            _text += self._str_basis_pol(_basis, self.source_dim)
        if is_latex_closure:
            _text += r' \)'
        return _text
    def str_destination_basis(self, is_latex_closure=True):
        if is_latex_closure:
            _text = r'\( '
        else:
            _text = ''
        if self.destination_domain == 'numeric':
            _text += self._str_basis_num(self.destination_basis, self.destination_dim)
        elif self.destination_domain == 'matrix':
            _text += self._str_basis_mat(self.destination_basis, self.destination_dim)
        else: # 'polynomial'       
            _text += self._str_basis_pol(self.destination_basis, self.destination_dim)
        if is_latex_closure:
            _text += r' \)'
        return _text


# In[ ]:


if __name__ == "__main__":
    for i in range(20):
        lm = LinearMap()
        lm.set_dimension_range(2,4)
        lm.generate(is_linear=True, is_source_standard_basis=False, is_destination_standard_basis=True)
        display([lm.source_domain, ' => ', lm.destination_domain])
        display([lm.map, ' in ', lm.representation])
        display(['DIM: ', lm.source_dim, ' => ', lm.destination_dim])
        display(['dim(IM)=', lm.image_dim, ' => dim(KER)=', lm.kernel_dim])
        display(['base(S)=', lm.source_basis, ' => base(D)=', lm.destination_basis])
        IPython.display.display(IPython.display.HTML(lm.str_source_basis()))
        IPython.display.display(IPython.display.HTML(lm.str_destination_basis()))
        display(lm.representation_matrix)
#        display(lm.data4nl)
#        display(lm.rref)
#        display(lm.matrix)
#        display(lm.source_basis)
#        display([lm.destination_basis, sympy.Matrix(lm.destination_basis).inverse_ADJ()])
#        display(lm.pivot_positions)
#        display(lm.id)        
        IPython.display.display(IPython.display.HTML(lm.str_map()))


# ## EigenSpace

# In[750]:


class EigenSpace():
    """
    This is an instance for generating an eigen space in several representations.
    """
    domain_type = ['numeric', 'matrix', 'polynomial'] # ns:()x, ne:(ax+), me: pe
    representation_type = ['symbolic', 'element-wise']
    def __init__(self, other=None):
        if other is not None:
            self.id = copy.deepcopy(other.id)
            self.dim_min = other.dim_min
            self.dim_max = other.dim_max
            self.elem_min = other.elem_min
            self.elem_max = other.elem_max
            self.root_min = other.root_min
            self.root_max = other.root_max
            self.dimension = other.dimension
            self.diagonalizable = other.diagonalizable
            self.domain = other.domain
            self.representation = other.representation
            self.eigen_space_dimensions = other.eigen_space_dimensions
            self.eigen_values = other.eigen_values
            self.pivot_positions = other.pivot_positions
            self.eigen_vectors = other.eigen_vectors
            self.representation_matrix = other.representation_matrix
        else:
            self.id = 0
            self.dim_min = 2
            self.dim_max = 4
            self.elem_min = -3
            self.elem_max =  3
            self.root_min = -3
            self.root_max =  3
            self.dimension = 2
            self.diagonalizable = True
            self.domain = self.domain_type[0]
            self.representation = self.representation_type[0]
            self.eigen_space_dimensions = None
            self.eigen_values = None
            self.pivot_positions = None
            self.eigen_vectors = None
            self.representation_matrix = None
    def __eq__(self, other):
        if not isinstance(other, EigenSpace):
            return False
        elif self.dimension != other.dimension or self.diagonalizable != other.diagonalizable:
            return False
        elif self.domain != other.domain or self.representation != other.representation:
            return False
        elif self.eigen_space_dimensions != other.eigen_space_dimensions or self.eigen_values != other.eigen_values:
            return False
        elif self.pivot_positions != other.pivot_positions or self.eigen_vectors != other.eigen_vectors:
            return False
        elif self.representation_matrix != other.representation_matrix:
            return False
        return True
    def __hash__(self):
        self.id = hash(str(self.dimension) + str(self.diagonalizable) + str(self.domain) + 
                       str(self.representation) + str(self.eigen_space_dimensions) + str(self.eigen_values) + 
                       str(self.pivot_positions) + str(self.eigen_vectors) + str(self.representation_matrix))
        return self.id
    def set_dimension_range(self, dmin, dmax):
        """
        The range of dimension of the ground vector space or length.
        """
        self.dim_min = dmin
        self.dim_max = dmax
    def set_element_range(self, emin, emax):
        """
        The range of elements of generators and other helper vectors.
        """
        self.elem_min = emin
        self.elem_max = emax
    def set_root_range(self, rmin, rmax):
        """
        The range of roots of characteristic polynomial (eigen values).
        """
        self.root_min = rmin
        self.root_max = rmax
    def generate(self, is_force_diagonalizable=None, is_force_symbolic=False, is_force_numeric=False, is_force_polynomial=False):
        self.generate_config(is_force_diagonalizable, is_force_symbolic, is_force_numeric, is_force_polynomial)
        self.generate_eigen_values()
        self.generate_eigen_vectors()
        self.generate_representation_matrix()
    def generate_config(self, is_force_diagonalizable=None, is_force_symbolic=False, is_force_numeric=False, is_force_polynomial=False):
        self.dimension = random.randint(self.dim_min, self.dim_max)
        if is_force_diagonalizable is None:
            if random.random() <= 0.5:
                self.diagonalizable = True
            else:
                self.diagonalizable = False
        else:
            self.diagonalizable = is_force_diagonalizable
        if is_force_symbolic:
            self.representation = 'symbolic'
        elif is_force_polynomial:
            self.representation = 'element-wise'
        else:
            self.representation = random.choice(self.representation_type)
        if is_force_numeric or self.representation == 'symbolic':
            self.domain = 'numeric'
        elif is_force_polynomial:
            self.domain = 'polynomial'
        else:
            self.domain = random.choice(self.domain_type)
        if integer_to_quasi_square(self.dimension)[0] == 1 and self.domain == 'matrix':
            self.domain = 'numeric'            
    def generate_eigen_values(self):
        self.eigen_space_dimensions = []
        _d = self.dimension
        while _d > 0:
            self.eigen_space_dimensions.append(random.randint(1,_d))
            _d -= self.eigen_space_dimensions[-1]
        if max(self.eigen_space_dimensions) == 1 and not self.diagonalizable:
            self.generate_eigen_values()
        elif len(self.eigen_space_dimensions) == 1 and self.dimension > 1 and self.diagonalizable:
            self.generate_eigen_values()
        else:
            _r = list(range(self.root_min, self.root_max+1))
            self.eigen_values = []
            for i in self.eigen_space_dimensions:
                self.eigen_values.append(random.choice(_r))
                _r = list(set(_r) - set(self.eigen_values))
    def generate_eigen_vectors(self):
        _random_positions = random.sample(range(self.dimension), self.dimension) 
        self.pivot_positions = []
        _candidate_pivs = []
        for d in self.eigen_space_dimensions:
            _pivot_positions = sorted(_random_positions[:d])
            _random_positions = _random_positions[d:]
            self.pivot_positions.append(_pivot_positions)
            if self.dimension - 1 not in _pivot_positions:
                _candidate_pivs = _candidate_pivs + _pivot_positions
        if len(_candidate_pivs) == 0:
            _candidate_pivs.append(self.dimension)
        _place_bottom_piv = random.choice(_candidate_pivs)
        self.eigen_vectors = []
        for esidx in range(len(self.pivot_positions)):
            es = self.pivot_positions[esidx]
            _eigen_vectors = []
            for imaxidx in range(len(es)):
                imax = es[imaxidx]
                _vec = [0 for j in range(imax)] + [1] + [0 for j in range(imax+1,self.dimension)]
                for i in range(imax):
                    if i not in es and i != _place_bottom_piv:
                        _vec[i] = random.randint(self.elem_min, self.elem_max)
                if imax == _place_bottom_piv:
                    _vec[-1] = 1
                    self.pivot_positions[esidx][imaxidx] = self.dimension - 1
                _eigen_vectors.append(_vec)
            self.eigen_vectors.append(_eigen_vectors)
    def generate_representation_matrix(self):
        _matrix_p = sympy.Matrix(flatten_list(self.eigen_vectors)).transpose()
        _matrix_d = sympy.zeros(self.dimension, self.dimension)
        _col_num = 0
        for idx in range(len(self.eigen_space_dimensions)):
            for i in range(self.eigen_space_dimensions[idx]):
                _matrix_d[_col_num,_col_num] = self.eigen_values[idx]
                if i > 0 and not self.diagonalizable:
                    _matrix_d[_col_num - 1,_col_num] = 1
                _col_num += 1
        _matrix_a = _matrix_p*_matrix_d*(_matrix_p.inv())
        self.representation_matrix = sympy.matrix2numpy(_matrix_a).tolist()
    def is_an_eigen_value(self, v):
        return v in self.eigen_values
    def is_an_eigen_vector_for_eigen_value(self, vec, v):
        if len(vec) != self.dimension or not self.is_an_eigen_value(v):
            return False
        if sum([abs(e) for e in vec]) == 0:
            return False
        _eigen_vecs = self.eigen_vectors[self.eigen_values.index(v)]
        return sympy.Matrix(_eigen_vecs + [vec]).rank() == len(_eigen_vecs)
    def is_an_eigen_vector(self, vec):
        for ev in self.eigen_values:
            if self.is_an_eigen_vector_for_eigen_value(vec, ev):
                return True
        return False
    def characteristic_polynomial(self, x):
        return sympy.expand((x*sympy.eye(self.dimension) - sympy.Matrix(self.representation_matrix)).det())
    def str_matrix(self, is_latex_closure=True):
        """
        This generates a latex expression of the representation matrix.
        """
        if is_latex_closure:
            _text = r'\( '
        else:
            _text = ''
        _text += sympy.latex(sympy.Matrix(self.representation_matrix), mat_delim='', mat_str='pmatrix')
        if is_latex_closure:
            _text += r' \)'
        return _text
    def _str_domain(self, dom, dim, is_latex_closure=False):
        if is_latex_closure:
            _text = r'\( '
        else:
            _text = ''
        if dom == 'numeric':
            _text += r'\mathbb{R}^{' + str(dim) + r'}'
        elif dom == 'matrix':
            sd = integer_to_quasi_square(dim) # sd = [r,c]
            _text += r'\mathbb{R}^{' + str(sd[0]) + r'\times ' + str(sd[1]) + r'}'
        else: # 'polynomial'
            _text += r'\mathbb{R}[x]_{' + str(dim - 1) + r'}'
        if is_latex_closure:
            _text += r' \)'
        return _text
    def _str_source_element_num(self):
        if self.representation == 'symbolic':
            return r'\vec{x}'
        _text = r'\begin{pmatrix}'
        for i in range(self.dimension):
            _text += r'x_{' + str(i+1) + r'} \\'    
        _text += r'\end{pmatrix}'
        return _text
    def _str_source_element_mat(self):
        if self.representation == 'symbolic':
            return r'M'
        _text = '' 
        sd = integer_to_quasi_square(self.dimension) # sd = [r,c]
        _text = r'\begin{pmatrix}'
        for i in range(sd[0]):
            for j in range(sd[1]):
                _text += r'm_{' + str(i*sd[1]+j+1) + r'}'    
            if j < sd[1] - 1:
                _text += r'&'
            else:
                _text += r'\\'
        _text += r'\end{pmatrix}'
        return _text
    def _str_source_element_pol(self):
        _text = ''
        for i in range(self.dimension - 1, -1, -1):
            if i < self.dimension - 1:
                _text += r'+'
            if i > 1:
                _text += r'a_{' + str(i+1) + r'}x^{' + str(i) + r'}'    
            elif i > 0:
                _text += r'a_{' + str(i+1) + r'}x'   
            else:
                _text += r'a_{' + str(i+1) + r'}'   
        return _text    
    def str_source_element(self, is_latex_closure=False):
        if is_latex_closure:
            _text = r'\( '
        else:
            _text = ''
        if self.domain == 'numeric':
            _text += self._str_source_element_num()
        elif self.domain == 'matrix':
            _text += self._str_source_element_mat()
        else: # 'polynomial'       
            _text += self._str_source_element_pol()
        if is_latex_closure:
            _text += r' \)'
        return _text  
    def _str_destination_element_num(self):
        _text = ''
        if self.representation == 'symbolic':
            _text += self.str_matrix(is_latex_closure=False) + r'\vec{x}'            
        else: # 'element-wise'
            _matrix = sympy.Matrix(self.representation_matrix)
            _vars = []
            if self.domain == 'numeric':
                for i in range(self.dimension):
                    _vars.append(sympy.Symbol('x'+str(i+1)))
            elif self.domain == 'matrix':
                sd = integer_to_quasi_square(self.dimension) # sd = [r,c]
                for i in range(sd[0]):
                    for j in range(sd[1]):
                        _vars.append(sympy.Symbol('m'+str(i*sd[1]+j+1)))
            else: # 'polynomial'
                for i in range(self.dimension):
                    _vars.append(sympy.Symbol('a'+str(i+1)))
            _vector = sympy.Matrix([[v] for v in _vars])
            _text += sympy.latex(_matrix*_vector, mat_delim='', mat_str='pmatrix')      
        return _text
    def _str_destination_element_mat(self):
        _text = ''
        sd = integer_to_quasi_square(self.dimension) # sd = [r,c]
        _matrix = sympy.Matrix(self.representation_matrix)
        _vars = []
        if self.domain == 'numeric':
            for i in range(self.dimension):
                _vars.append(sympy.Symbol('x'+str(i+1)))
        elif self.domain == 'matrix':
            source_sd = integer_to_quasi_square(self.dimension) # sd = [r,c]
            for i in range(source_sd[0]):
                for j in range(source_sd[1]):
                    _vars.append(sympy.Symbol('m'+str(i*source_sd[1]+j+1)))
        else: # 'polynomial'
            for i in range(self.dimension):
                _vars.append(sympy.Symbol('a'+str(i+1)))
        _vector = sympy.Matrix([[v] for v in _vars])
        _result = _matrix*_vector
        _text += sympy.latex(sympy.Matrix(partition_list([e for e in _result], sd[1])), mat_delim='', mat_str='pmatrix')      
        return _text
    def _str_destination_element_pol(self):
        _text = ''
        _matrix = sympy.Matrix(self.representation_matrix)
        _vars = []
        if self.domain == 'numeric':
            for i in range(self.dimension):
                _vars.append(sympy.Symbol('x'+str(i+1)))
        elif self.domain == 'matrix':
            sd = integer_to_quasi_square(self.dimension) # sd = [r,c]
            for i in range(sd[0]):
                for j in range(sd[1]):
                    _vars.append(sympy.Symbol('m'+str(i*sd[1]+j+1)))
        else: # 'polynomial'
            for i in range(self.dimension):
                _vars.append(sympy.Symbol('a'+str(i+1)))
        _vector = sympy.Matrix([[v] for v in _vars])
        _result = _matrix*_vector   
        _x = sympy.Symbol('x')
        _rmatrix = sympy.Matrix([[_x**i for i in range(0, self.dimension)]])*_result
        _text += sympy.latex(_rmatrix[0,0])
        return _text    
    def str_destination_element(self, is_latex_closure=False):
        if is_latex_closure:
            _text = r'\( '
        else:
            _text = ''
        if self.domain == 'numeric':
            _text += self._str_destination_element_num()
        elif self.domain == 'matrix':
            _text += self._str_destination_element_mat()
        else: # 'polynomial'       
            _text += self._str_destination_element_pol()
        if is_latex_closure:
            _text += r' \)'
        return _text
    def str_map(self, is_latex_closure=True):
        """
        This generates a latex expression of the map.
        """
        if is_latex_closure:
            _text = r'\( '
        else:
            _text = ''
        _text += r'f:' + self._str_domain(self.domain, self.dimension)
        _text += r'\rightarrow ' + self._str_domain(self.domain, self.dimension)
        _text += r',\;'
        _text += self.str_source_element()
        _text += r'\mapsto '
        _text += self.str_destination_element()
        if is_latex_closure:
            _text += r' \)'
        return _text
    def str_vector(self, vec, is_latex_closure=False):
        if is_latex_closure:
            _text = r'\( '
        else:
            _text = ''
        if self.domain == 'numeric':
            _text += sympy.latex(sympy.Matrix([vec]).transpose(), mat_delim='', mat_str='pmatrix')
        elif self.domain == 'matrix':
            _text += sympy.latex(sympy.Matrix(partition_list(vec,integer_to_quasi_square(len(vec))[1])), mat_delim='', mat_str='pmatrix')
        else: # 'polynomial'
            _x = sympy.Symbol('x')
            _text += sympy.latex(sum([vec[i]*_x**i for i in range(len(vec))]))
        if is_latex_closure:
            _text += r' \)'
        return _text
    def _str_eigen_space_num(self, vecs):
        _text = r'\left\{\begin{array}{c|c}'
        for i in range(len(vecs)):
            _text += r'c_{' + str(i+1) + r'}'
            _text += self.str_vector(vecs[i])
            if i < len(vecs) - 1:
                _text += r'+'
        _text += r'&'
        for i in range(len(vecs)):
            _text += r'c_{' + str(i+1) + r'}'
            if i < len(vecs) - 1:
                _text += r','
        _text += r'\in\mathbb{R}'
        _text += r'\end{array}\right\}'
        return _text
    def _str_eigen_space_mat(self, vecs):
        _text = r'\left\{\begin{array}{c|c}'
        for i in range(len(vecs)):
            _text += r'c_{' + str(i+1) + r'}'
            _text += self.str_vector(vecs[i])
            if i < len(vecs) - 1:
                _text += r'+'
        _text += r'&'
        for i in range(len(vecs)):
            _text += r'c_{' + str(i+1) + r'}'
            if i < len(vecs) - 1:
                _text += r','
        _text += r'\in\mathbb{R}'
        _text += r'\end{array}\right\}'
        return _text
    def _str_eigen_space_pol(self, vecs):
        _text = r'\left\{\begin{array}{c|c}'
        for i in range(len(vecs)):
            _text += r'c_{' + str(i+1) + r'}'
            _text += r'(' + self.str_vector(vecs[i]) + r')'
            if i < len(vecs) - 1:
                _text += r'+'
        _text += r'&'
        for i in range(len(vecs)):
            _text += r'c_{' + str(i+1) + r'}'
            if i < len(vecs) - 1:
                _text += r','
        _text += r'\in\mathbb{R}'
        _text += r'\end{array}\right\}'
        return _text
    def str_eigen_space_for_the_given(self, v, vecs, is_diagonalizable, domain, is_latex_closure=False):
        if is_latex_closure:
            _text = r'\( '
        else:
            _text = ''
        _text += r'W\left(' + sympy.latex(v) + r';f\right)='
        if not is_diagonalizable:
            vecs = [vecs[0]]
        if domain == 'numeric':
            _text += self._str_eigen_space_num(vecs)
        elif domain == 'matrix':
            _text += self._str_eigen_space_mat(vecs)
        else: # 'polynomial'       
            _text += self._str_eigen_space_pol(vecs)                
        if is_latex_closure:
            _text += r' \)'
        return _text
    def str_eigen_space(self, v, is_latex_closure=False):
        """
        This generates a latex expression of the eigen space of the given eigen value.
        """
        return self.str_eigen_space_for_the_given(v, self.eigen_vectors[self.eigen_values.index(v)], self.diagonalizable, self.domain, is_latex_closure)
    def str_eigen_spaces_for_the_given(self, vs, vecss, is_diagonalizable, domain, is_latex_closure=True):
        if is_latex_closure:
            _text = r'\( '
        else:
            _text = ''
        for i in range(len(vs)):
            _text += self.str_eigen_space_for_the_given(vs[i], vecss[i], is_diagonalizable, domain, is_latex_closure=False)
            if i < len(vs) - 1:
                _text += r',\;'
        if is_latex_closure:
            _text += r' \)'
        return _text
    def str_eigen_spaces(self, is_latex_closure=True):
        """
        This generates a latex expression of the eigen spaces.
        """
        return self.str_eigen_spaces_for_the_given(self.eigen_values, self.eigen_vectors, self.diagonalizable, self.domain, is_latex_closure)


# In[ ]:


if __name__ == "__main__":
    for i in range(20):
        es = EigenSpace()
        es.set_dimension_range(2,3)
        es.generate_config()
        es.generate_eigen_values()
        es.generate_eigen_vectors()
        es.generate_representation_matrix()
        display([es.dimension, es.diagonalizable, es.representation, es.domain])
        display([es.eigen_space_dimensions, es.eigen_values])
        display([es.pivot_positions, es.eigen_vectors])
        display(sympy.Matrix(es.representation_matrix))
        display(sympy.Matrix(es.representation_matrix).eigenvects())
        IPython.display.display(IPython.display.HTML('<div>' + es.str_map(is_latex_closure=True) + '</div>'))
        IPython.display.display(IPython.display.HTML('<div>' + es.str_eigen_spaces(is_latex_closure=True) + '</div>'))


# ## Inner space

# In[12]:


class InnerSpace():
    """
    This generates an inner space satisfying the specified condition.
    domain is in {'Cn','Rn', 'aRn', 'Rnm', 'Rx', None}.
    For symbolic expressions, ['vector', index], ['inner_product', vecA, vecB], 
    ['norm', vecA], ['addition', vecA, vecB], ['subtraction', vecA, vecB], ['multiplication', alpha, vecA]
    """
    def __init__(self, domain):
        self.domain = domain
    def str_domain(self, is_latex_closure=True):
        if is_latex_closure:
            _text = r'\( '
        else:
            _text = ''
        if self.domain == 'Cn':
            _text += r'\mathbb{C}^{n}'
        elif self.domain == 'Rn':
            _text += r'\mathbb{R}^{n}'
        elif self.domain == 'aRn':
            _text += r'\mathbb{R}^{n}'
        elif self.domain == 'Rnm':
            _text += r'\mathbb{R}^{n\times m}'
        elif self.domain == 'Rx':
            _text += r'\mathbb{R}[x]_{n}'
        else:
            _text += r'V'
        if is_latex_closure:
            _text += r' \)'
        return _text  
    def str_inner_product(self, is_latex_closure=True):
        if is_latex_closure:
            _text = r'\( '
        else:
            _text = ''
        if self.domain == 'Cn':
            _text += r'\vec{a}=(a_i),\vec{b}=(b_i)\in\mathbb{C}^{n},\;(\vec{a},\vec{b}):=\sum_{i=1}^{n}a_i\bar{b_i}'
        elif self.domain == 'Rn':
            _text += r'\vec{a}=(a_i),\vec{b}=(b_i)\in\mathbb{R}^{n},\;(\vec{a},\vec{b}):=\sum_{i=1}^{n}a_ib_i'
        elif self.domain == 'aRn':
            _text += r'\vec{a}=(a_i),\vec{b}=(b_i)\in\mathbb{R}^{n},\;(\vec{a},\vec{b}):=\sum_{i=1}^{n}i\times a_ib_i'
        elif self.domain == 'Rnm':
            _text += r'A,B\in\mathbb{R}^{n\times m},\;(A,B):=\textrm{tr}({}^{t}\!BA)'
        elif self.domain == 'Rx':
            _text += r'f(x),g(x)\in\mathbb{R}[x]_{n},\;(f,g):=\int_{-1}^{1}f(x)g(x)dx'
        else:
            _text += r'\vec{a},\vec{b}\in V,\;(\vec{a},\vec{b})'
        if is_latex_closure:
            _text += r' \)'
        return _text  
    def str_vector(self, vec, is_latex_closure=False):
        if is_latex_closure:
            _text = r'\( '
        else:
            _text = ''
        if self.domain == 'Cn':
            _text += sympy.latex(sympy.Matrix([vec]).transpose(), mat_delim='', mat_str='pmatrix')
        elif self.domain == 'Rn':
            _text += sympy.latex(sympy.Matrix([vec]).transpose(), mat_delim='', mat_str='pmatrix')
        elif self.domain == 'aRn':
            _text += sympy.latex(sympy.Matrix([vec]).transpose(), mat_delim='', mat_str='pmatrix')
        elif self.domain == 'Rnm':
            _text += sympy.latex(sympy.Matrix(partition_list(vec,integer_to_quasi_square(len(vec))[1])), mat_delim='', mat_str='pmatrix')
        elif self.domain == 'Rx':
            _x = sympy.Symbol('x')
#            _text += sympy.latex(sum([vec[i]*_x**i for i in range(len(vec))]), long_frac_ratio=1)
            _head = True
            for i in reversed(range(len(vec))):
                if vec[i] == 1:
                    if _head:
                        _text += sympy.latex(vec[i]*_x**i)
                    else:
                        _text += r'+' + sympy.latex(vec[i]*_x**i)
                    _head = False
                elif vec[i] == -1:
                    _text += sympy.latex(vec[i]*_x**i)
                    _head = False
                elif vec[i] > 0:
                    if _head:
                        _text += sympy.latex(vec[i],  long_frac_ratio=1) + (sympy.latex(_x**i) if i != 0 else r'')
                    else:
                        _text += r'+' + sympy.latex(vec[i],  long_frac_ratio=1) + (sympy.latex(_x**i) if i != 0 else r'')
                    _head = False
                elif vec[i] < 0:
                    _text += sympy.latex(vec[i],  long_frac_ratio=1) + (sympy.latex(_x**i) if i != 0 else r'')
                    _head = False
        if is_latex_closure:
            _text += r' \)'
        return _text
    def str_vectors(self, vecs, is_latex_closure=True, variable=None):
        if is_latex_closure:
            _text = r'\( '
        else:
            _text = ''
        for i in range(len(vecs)):
            if variable is not None:
                _text += r'\vec{' + variable + r'_{' + str(i+1) + r'}}='
            _text += self.str_vector(vecs[i], is_latex_closure=False)
            if i < len(vecs) - 1:
                _text += r',\;'
        if is_latex_closure:
            _text += r' \)'
        return _text
    @classmethod
    def str_symbolic(cls, expr, is_latex_closure=True, variable=None, parenthesis=True):
        if variable is None:
            return cls.str_symbolic(expr, is_latex_closure, variable=r'v')
        if is_latex_closure:
            _text = r'\( '
        else:
            _text = ''
        if expr[0] == 'vector': # ['vector', index]
            _text += r'\vec{' + variable + r'_{' + str(expr[1]) + r'}}'
        elif expr[0] == 'inner_product': # ['inner_product', vecA, vecB]
            _text += r'\left(' + cls.str_symbolic(expr[1], is_latex_closure=False, variable=variable, parenthesis=False) + r',\;'+ cls.str_symbolic(expr[2], is_latex_closure=False, variable=variable, parenthesis=False) + r'\right)'
        elif expr[0] == 'norm': # ['norm', vecA]
            _text += r'\left\lVert ' + cls.str_symbolic(expr[1], is_latex_closure=False, variable=variable, parenthesis=False) + r'\right\rVert'
        elif expr[0] == 'addition': # ['addition', vecA, vecB]
            if parenthesis:
                _text += r'\left(' + cls.str_symbolic(expr[1], is_latex_closure=False, variable=variable, parenthesis=False) + r'+'+ cls.str_symbolic(expr[2], is_latex_closure=False, variable=variable, parenthesis=False) + r'\right)'
            else:
                _text += cls.str_symbolic(expr[1], is_latex_closure=False, variable=variable, parenthesis=False) + r'+'+ cls.str_symbolic(expr[2], is_latex_closure=False, variable=variable, parenthesis=False)
        elif expr[0] == 'subtraction': # ['subtraction', vecA, vecB]
            if parenthesis:
                _text += r'\left(' + cls.str_symbolic(expr[1], is_latex_closure=False, variable=variable, parenthesis=False) + r'-'+ cls.str_symbolic(expr[2], is_latex_closure=False, variable=variable) + r'\right)'
            else:
                _text += cls.str_symbolic(expr[1], is_latex_closure=False, variable=variable, parenthesis=False) + r'-'+ cls.str_symbolic(expr[2], is_latex_closure=False, variable=variable)
        elif expr[0] == 'multiplication': # ['multiplication', alpha, vecA]
            if expr[1] < 0:
                _text += r'\left(' + sympy.latex(expr[1]) + r'\right)' + cls.str_symbolic(expr[2], is_latex_closure=False, variable=variable)
            else:
                _text += sympy.latex(expr[1]) + cls.str_symbolic(expr[2], is_latex_closure=False, variable=variable)
        if is_latex_closure:
            _text += r' \)'
        return _text
    @classmethod
    def generate_symbolic_sum(cls, dim, emin=-3, emax=3):
        coefs = [0 for i in range(dim)]
        while sum([0 if e == 0 else 1 for e in coefs]) <= 1:
            coefs = [random.randint(emin, emax) for i in range(dim)]
        vecs = []
        for i in range(dim):
            if coefs[i] == 1:
                vecs.append(['vector', i+1])
            elif coefs[i] == -1:
                vecs.append(['sub_vector', i+1])
            elif coefs[i] != 0:
                vecs.append(['multiplication', coefs[i], ['vector', i+1]])
        expr = vecs[0] if vecs[0][0] != 'sub_vector' else ['vector', vecs[0][1]]
        for v in vecs[1:]:
            if v[0] == 'vector':
                expr = ['addition', expr, v]
            elif v[0] == 'sub_vector':
                expr = ['subtraction', expr, ['vector', v[1]]]
            elif v[1] > 0:
                expr = ['addition', expr, v]
            else:
                expr = ['subtraction', expr, [v[0], -v[1], v[2]]]
        return expr
    @classmethod
    def generate_symbolic_norm(cls, dim, emin=-3, emax=3):
        return ['norm', cls.generate_symbolic_sum(dim, emin, emax)]
    @classmethod
    def generate_symbolic_inner_product(cls, dim, emin=-3, emax=3):
        return ['inner_product', cls.generate_symbolic_sum(dim, emin, emax), cls.generate_symbolic_sum(dim, emin, emax)]
    @classmethod
    def generate_symbolic(cls, dim, emin=-3, emax=3, norm_ratio=0.5):
        if random.random() < norm_ratio:
            return cls.generate_symbolic_norm(dim, emin, emax)
        else:
            return cls.generate_symbolic_inner_product(dim, emin, emax)
    @classmethod
    def expand_symbolic(cls, expr, wrong=False, nest=-1):
        if nest == 0:
            return expr
        elif expr[0] == 'vector':
            return expr
        elif expr[0] == 'addition' or expr[0] == 'subtraction':
            return [expr[0], cls.expand_symbolic(expr[1], wrong=wrong, nest=nest-1), cls.expand_symbolic(expr[2], wrong=wrong, nest=nest-1)]
        elif expr[0] == 'multiplication':
            if wrong and random.random() < 0.25:
                return cls.expand_symbolic(expr[2], wrong=wrong, nest=nest-1)
            return [expr[0], expr[1], cls.expand_symbolic(expr[2], wrong=wrong, nest=nest-1)]
        elif expr[0] == 'inner_product': # ['inner_product', vecA, vecB]
            if expr[1][0] == 'multiplication':
                if wrong and random.random() < 0.25:
                    return cls.expand_symbolic(['inner_product', expr[1][2], expr[2]], wrong=wrong, nest=nest-1)
                return ['multiplication', expr[1][1], cls.expand_symbolic(['inner_product', expr[1][2], expr[2]], wrong=wrong, nest=nest-1)]
            elif expr[2][0] == 'multiplication':
                if wrong and random.random() < 0.25:
                    return cls.expand_symbolic(['inner_product', expr[1], expr[2][2]], wrong=wrong, nest=nest-1)
                return ['multiplication', expr[2][1], cls.expand_symbolic(['inner_product', expr[1], expr[2][2]], wrong=wrong, nest=nest-1)]
            elif expr[1][0] == 'addition':
                if wrong and random.random() < 0.25:
                    return cls.expand_symbolic(['inner_product', expr[1][1], expr[2]], wrong=wrong, nest=nest-1)
                return ['addition', cls.expand_symbolic(['inner_product', expr[1][1], expr[2]], wrong=wrong, nest=nest-1), cls.expand_symbolic(['inner_product', expr[1][2], expr[2]], wrong=wrong, nest=nest-1)]
            elif expr[2][0] == 'addition':
                if wrong and random.random() < 0.25:
                    return cls.expand_symbolic(['inner_product', expr[1], expr[2][1]], wrong=wrong, nest=nest-1)
                return ['addition', cls.expand_symbolic(['inner_product', expr[1], expr[2][1]], wrong=wrong, nest=nest-1), cls.expand_symbolic(['inner_product', expr[1], expr[2][2]], wrong=wrong, nest=nest-1)]
            elif expr[1][0] == 'subtraction':
                if wrong and random.random() < 0.25:
                    return cls.expand_symbolic(['inner_product', expr[1][1], expr[2]], wrong=wrong, nest=nest-1)
                return ['subtraction', cls.expand_symbolic(['inner_product', expr[1][1], expr[2]], wrong=wrong, nest=nest-1), cls.expand_symbolic(['inner_product', expr[1][2], expr[2]], wrong=wrong, nest=nest-1)]
            elif expr[2][0] == 'subtraction':
                if wrong and random.random() < 0.25:
                    return cls.expand_symbolic(['inner_product', expr[1], expr[2][1]], wrong=wrong, nest=nest-1)
                return ['subtraction', cls.expand_symbolic(['inner_product', expr[1], expr[2][1]], wrong=wrong, nest=nest-1), cls.expand_symbolic(['inner_product', expr[1], expr[2][2]], wrong=wrong, nest=nest-1)]
            else:
                return expr
        elif expr[0] == 'norm': # ['norm', vecA]
            if expr[1][0] == 'multiplication':
                if wrong and random.random() < 0.25:
                    return cls.expand_symbolic(['norm', expr[1][2], expr[1][2]], wrong=wrong, nest=nest-1)
                return ['multiplication', expr[1][1], cls.expand_symbolic(['norm', expr[1][2], expr[1][2]], wrong=wrong, nest=nest-1)]
            else:
                return expr
        return expr
    @classmethod
    def eval_symbolic(cls, expr, inners, wrong=False):
        if wrong:
            return cls._eval_symbolic(cls.expand_symbolic(expr, wrong=True), inners, wrong=True)
        return cls._eval_symbolic(cls.expand_symbolic(expr), inners)
    @classmethod
    def _eval_symbolic(cls, expr, inners, wrong=False):
        if expr[0] == 'inner_product': # ['inner_product', vecA, vecB]
            return inners[expr[1][1]-1][expr[2][1]-1]
        elif expr[0] == 'norm': # ['norm', vecA]
            _mid_output = cls.eval_symbolic(['inner_product', expr[1], expr[1]], inners, wrong=wrong)
            if wrong and _mid_output < 0:
                _mid_output *= -1
            return sympy.sqrt(_mid_output)
        elif expr[0] == 'addition': # ['addition', vecA, vecB]
            return cls._eval_symbolic(expr[1], inners) + cls._eval_symbolic(expr[2], inners)
        elif expr[0] == 'subtraction': # ['subtraction', vecA, vecB]
            return cls._eval_symbolic(expr[1], inners) - cls._eval_symbolic(expr[2], inners)
        elif expr[0] == 'multiplication': # ['multiplication', alpha, vecA]
            return expr[1] * cls._eval_symbolic(expr[2], inners)
        raise ValueError
    def inner_product(self, vecA, vecB):
        if self.domain == 'Cn':
            return self._inner_product_Cn(vecA, vecB)
        elif self.domain == 'Rn':
            return self._inner_product_Rn(vecA, vecB)
        elif self.domain == 'aRn':
            return self._inner_product_aRn(vecA, vecB)
        elif self.domain == 'Rnm':
            return self._inner_product_Rnm(vecA, vecB)
        elif self.domain == 'Rx':
            return self._inner_product_Rx(vecA, vecB)
        else:
            return ['inner_product', vecA, vecB]
    @classmethod
    def _inner_product_Cn(cls, vecA, vecB):
        return sum(vecA[i]*sympy.conjugate(vecB[i]) for i in range(len(vecA)))
    @classmethod
    def _inner_product_Rn(cls, vecA, vecB):
        return sum(vecA[i]*vecB[i] for i in range(len(vecA)))
    @classmethod
    def _inner_product_aRn(cls, vecA, vecB):
        return sum((i+1)*vecA[i]*vecB[i] for i in range(len(vecA)))
    @classmethod
    def _inner_product_Rnm(cls, vecA, vecB):
        return sum(vecA[i]*vecB[i] for i in range(len(vecA)))        
    @classmethod
    def _inner_product_Rx(cls, vecA, vecB):
        _x = sympy.Symbol('x')
        polyA = sum([vecA[i]*_x**i for i in range(len(vecA))])
        polyB = sum([vecB[i]*_x**i for i in range(len(vecB))])
        return sympy.integrate(polyA*polyB, (_x, -1, 1))
    def norm(self, vecA):
        if self.domain is None:
            return ['norm', vecA]
        else:
            return sympy.sqrt(self.inner_product(vecA, vecA))
    def orthogonalize(self, vecs, n=None):
        if n is None or n >= len(vecs):
            n = len(vecs)
        elif n < 0:
            n = len(vecs) + n + 1
        return self._orthogonalize(vecs, n)
    def _orthogonalize(self, vecs, n):
        orthogonalized_vecs = []
        if n > 0:
            orthogonalized_vecs.append(vecs[0])
        for i in range(1,n):
            _vec = vecs[i]
            for v in orthogonalized_vecs:
                _mu = self.inner_product(vecs[i], v) * sympy.Pow(self.inner_product(v, v), -1)
                _vec = [_vec[j] - _mu*v[j] for j in range(len(v))]
            orthogonalized_vecs.append(_vec)
        return orthogonalized_vecs
    def normalize(self, vecs):
        normalized_vecs = []
        for v in vecs:
            _norm_reciprocal = sympy.Pow(self.norm(v), -1)
            _vec = [e*_norm_reciprocal for e in v]
            normalized_vecs.append(_vec)
        return normalized_vecs
    def orthonomalize(self, vecs):
        _orthogonalized_vecs = self.orthogonalize(vecs)
        return self.normalize(_orthogonalized_vecs)


# In[705]:


if __name__ == "__main__":
    ins = InnerSpace(domain='Rx')
    vecA = [1,1,1]
    vecB = [1,1,0]
    vecC = [1,0,0]
    IPython.display.display(IPython.display.HTML(ins.str_vectors([vecA,vecB,vecC], variable=r'v')))
    display(ins.inner_product(vecA, vecB))
    display(ins.norm(vecA))
    IPython.display.display(IPython.display.HTML(ins.str_domain()))
    IPython.display.display(IPython.display.HTML(ins.str_inner_product()))
    for v in ins.orthogonalize([vecA,vecB,vecC], -2):
        display(v)
    vecs = ins.orthonomalize([vecA,vecB,vecC])
    IPython.display.display(IPython.display.HTML(ins.str_vectors(vecs, variable=r'v')))
    ins = InnerSpace(domain='Rn')
    IPython.display.display(IPython.display.HTML(ins.str_vectors(vecs, variable=r'v')))


# In[631]:


if __name__ == "__main__":
    ins = InnerSpace(domain=None)
    expr = ['multiplication', 2, ['inner_product', ['addition', ['vector', 2], ['multiplication', -3, ['subtraction', ['vector', 3], ['vector', 4]]]], ['vector', 1]]]
    IPython.display.display(IPython.display.HTML(ins.str_symbolic(expr, variable=r'v')))
    IPython.display.display(IPython.display.HTML(ins.str_symbolic(ins.expand_symbolic(expr), variable=r'v')))
    display(ins.eval_symbolic(expr, [[1,2,3,4],[2,3,4,1],[3,4,1,2],[4,1,2,3]]))
    expr = ins.generate_symbolic(2, -3, 3)
    print(expr)
    IPython.display.display(IPython.display.HTML(ins.str_symbolic(expr, variable=r'v')))
    IPython.display.display(IPython.display.HTML(ins.str_symbolic(ins.expand_symbolic(expr), variable=r'v')))
    display(ins.eval_symbolic(expr, [[1,2,3,4],[2,3,4,1],[3,4,1,2],[4,1,2,3]]))


# In[ ]:




