#!/usr/bin/env python
# coding: utf-8

# Moodle XML Question Generator, (C) 2019-2020 Kosaku Nagasaka (Kobe University)

# ## モジュールファイルの生成

# In[14]:


if __name__ == "__main__":
    import subprocess
    subprocess.run(['jupyter', 'nbconvert', '--to', 'python', '_linear_algebra_next_generation.ipynb','--output','linear_algebra_next_generation.py'])


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


# In[15]:


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


# In[17]:


if __name__ == "__main__":
    print(thread_list([[[1,2],[3,4]],[[5,6],[7,8]],[[10,11,12],[13,14,15]]]))


# In[18]:


def sometimes_zero_randint(imin, imax, zratio):
    if random.random() < zratio:
        return 0
    elif 0 < imin or imax < 0:
        return random.randint(imin, imax)
    else:
        return random.choice(list(set(range(imin,imax+1)).difference(set([0]))))


# In[19]:


if __name__ == "__main__":
    print([sometimes_zero_randint(-1,1,0.5) for i in range(20)])


# In[4]:


def nonzero_randint(imin, imax):
    if 0 < imin or imax < 0:
        return random.randint(imin, imax)
    else:
        return random.choice(list(set(range(imin,imax+1)).difference(set([0]))))


# In[21]:


if __name__ == "__main__":
    print([nonzero_randint(-1,1) for i in range(20)])


# In[22]:


def nonone_randint(imin, imax):
    if 1 < imin or imax < 1:
        return random.randint(imin, imax)
    else:
        return random.choice(list(set(range(imin,imax+1)).difference(set([1]))))


# In[23]:


if __name__ == "__main__":
    print([nonone_randint(0,3) for i in range(20)])


# In[11]:


def nonzeroone_randint(imin, imax):
    if 1 < imin or imax < -1:
        return random.randint(imin, imax)
    else:
        return random.choice(list(set(range(imin,imax+1)).difference(set([-1,0,1]))))


# In[13]:


if __name__ == "__main__":
    print([nonzeroone_randint(-3,3) for i in range(20)])


# In[5]:


def rationalized_vector(vec):
    _denom = [sympy.denom(ele) for ele in vec]
    if len(_denom) > 1:
        _dlcm = sympy.ilcm(*_denom)
    else:
        _dlcm = _denom[0]
    return [_dlcm*ele for ele in vec]


# In[25]:


if __name__ == "__main__":
    print(rationalized_vector([sympy.Rational(-3,7),-2,sympy.Rational(3,10)]))
    print(rationalized_vector([sympy.Rational(-3,7)]))


# In[6]:


def is_zero_vector(vec):
    for ele in vec:
        if ele != 0:
            return False
    return True


# In[27]:


if __name__ == "__main__":
    print(is_zero_vector([1,2,3]), is_zero_vector([0,0,0]))


# In[7]:


def is_zero_vectors(vecs):
    for vec in vecs:
        if not is_zero_vector(vec):
            return False
    return True


# In[29]:


if __name__ == "__main__":
    print(is_zero_vectors([[0,0,0],[1,2,3]]), is_zero_vectors([[0,0],[0,0,0]]))


# In[8]:


def is_integer_vector(vec):
    for ele in vec:
        if not sympy.sympify(ele).is_integer:
            return False
    return True


# In[31]:


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


# In[33]:


if __name__ == "__main__":
    print(integer_to_quasi_square(8))
    print(integer_to_quasi_square(7))


# In[10]:


def flatten_list(alist):
    rlist = []
    for lis in alist:
        rlist = rlist + lis
    return rlist


# In[35]:


if __name__ == "__main__":
    print(flatten_list([[[1,2,3],[4,5]],[1,2]]))


# In[36]:


def flatten_list_all(alist):
    if type(alist) is not list:
        return [alist]
    rlist = []
    for lis in alist:
        rlist = rlist + flatten_list_all(lis)
    return rlist


# In[37]:


if __name__ == "__main__":
    print(flatten_list_all([[[1,[2,3]],[[4],5]],[1,2]]))


# ## matrix computations

# In[270]:


class MatrixComputations():
    """
    This generates a general expression of vector and matrix.
    """
    _incorrect_types = ['forgot_rest', 'forgot_next_cols', 'forgot_next_rows', 'computation_miss', 'mul_instead_dot', 'mixed']
    _tex_scalars = [r'\alpha', r'\beta', r'\gamma']
    _tex_vectors = [r'\vec{x}', r'\vec{y}', r'\vec{z}', r'\vec{v}', r'\vec{w}']
    _tex_matrices = ['A', 'B', 'C', 'D', 'X', 'Y', 'Z']
    _reserved_words = ['mul', 'add', 'sub', 'scal', 'mat', 'def']
    # each skeleton = [defs, expr]
    # defs = [['mat', rows, cols, name], ['scal', name], ...]
    # expr = ['mul', expr0, expr1] (an example)
    _skeletons = dict()
    _skeletons["ms_22"] = [[[['scal','scalA'],['mat',2,2,'matA']], ['mul','scalA', 'matA']]]
    _skeletons["ms_mn"] = [[[['scal','scalA'],['mat','m','n','matA']], ['mul','scalA', 'matA']]]
    _skeletons["m_m_22"] = [[[['mat',2,2,'matA'], ['mat',2,2,'matB']], ['add','matA','matB']], 
                            [[['mat',2,2,'matA'], ['mat',2,2,'matB']], ['sub','matA','matB']]]
    _skeletons["m_m_mn"] = [[[['mat','m','n','matA'], ['mat','m','n','matB']], ['add','matA','matB']], 
                            [[['mat','m','n','matA'], ['mat','m','n','matB']], ['sub','matA','matB']]]
    _skeletons["mv_22"] = [[[['mat',2,2,'matA'], ['mat',2,1,'vecB']], ['mul','matA','vecB']]]
    _skeletons["mv_mn"] = [[[['mat','m','n','matA'], ['mat','n',1,'vecB']], ['mul','matA','vecB']]]
    _skeletons["mm_22"] = [[[['mat',2,2,'matA'], ['mat',2,2,'matB']], ['mul','matA','matB']]]
    _skeletons["mm_mn"] = [[[['mat','m','m','matA'], ['mat','m','m','matB']], ['mul','matA','matB']],
                           [[['mat','m','n','matA'], ['mat','n','k','matB']], ['mul','matA','matB']],
                           [[['mat','m',1,'matA'], ['mat',1,'k','matB']], ['mul','matA','matB']]]
    _skeletons["msvm_22"] = [[[['scal','scalA'],['scal','scalB'],['mat',2,2,'matA']], ['add',['mul','scalA','matA'],['mul','scalB','matA']]],
                             [[['scal','scalA'],['mat',2,2,'matA'],['mat',2,2,'matB']], ['add',['mul','scalA','matA'],['mul','scalA','matB']]],
                             [[['scal','scalA'],['mat',2,2,'matA'],['mat',2,1,'vecB'],['mat',2,1,'vecC']], ['add',['mul','matA','vecB'],['mul','scalA',['mul','matA','vecC']]]],
                             [[['scal','scalA'],['mat',2,2,'matA'],['mat',2,2,'matB'],['mat',2,1,'vecC']], ['add',['mul','matA','vecC'],['mul','scalA',['mul','matB','vecC']]]],
                             [[['mat',2,2,'matA'],['mat',2,2,'matB'],['mat',2,2,'matC']], ['add',['mul','matA','matB'],['mul','matA','matC']]],
                             [[['mat',2,2,'matA'],['mat',2,2,'matB'],['mat',2,2,'matC']], ['sub',['mul','matA','matB'],['mul','matA','matC']]],
                             [[['mat',2,2,'matA'],['mat',2,2,'matB'],['mat',2,2,'matC']], ['add',['mul','matA','matC'],['mul','matB','matC']]],
                             [[['mat',2,2,'matA'],['mat',2,2,'matB'],['mat',2,2,'matC']], ['sub',['mul','matA','matC'],['mul','matB','matC']]],
                             [[['mat',2,2,'matA'],['mat',2,2,'matB']], ['add',['mul','matA','matA'],['mul','matA','matB'],['mul','matB','matA'],['mul','matB','matB']]],
                             [[['mat',2,2,'matA'],['mat',2,2,'matB'],['def',2,'scalA']], ['add',['mul','matA','matA'],['mul','scalA',['mul','matA','matB']],['mul','matB','matB']]],
                             [[['mat',2,2,'matA'],['mat',2,2,'matB'],['mat',2,2,'matC'],['mat',2,1,'vecD']], ['mul',['mul',['mul','matA','matB'],'matC'],'vecD']]]
    _skeletons["msvm_mn"] = [[[['scal','scalA'],['scal','scalB'],['mat','m','n','matA']], ['add',['mul','scalA','matA'],['mul','scalB','matA']]],
                             [[['scal','scalA'],['mat','m','n','matA'],['mat','m','n','matB']], ['add',['mul','scalA','matA'],['mul','scalA','matB']]],
                             [[['scal','scalA'],['mat','m','n','matA'],['mat','n',1,'vecB'],['mat','n',1,'vecC']], ['add',['mul','matA','vecB'],['mul','scalA',['mul','matA','vecC']]]],
                             [[['scal','scalA'],['mat','m','n','matA'],['mat','m','n','matB'],['mat','n',1,'vecC']], ['add',['mul','matA','vecC'],['mul','scalA',['mul','matB','vecC']]]],
                             [[['mat','m','n','matA'],['mat','n','k','matB'],['mat','n','k','matC']], ['add',['mul','matA','matB'],['mul','matA','matC']]],
                             [[['mat','m','n','matA'],['mat','n','k','matB'],['mat','n','k','matC']], ['sub',['mul','matA','matB'],['mul','matA','matC']]],
                             [[['mat','m','n','matA'],['mat','m','n','matB'],['mat','n','k','matC']], ['add',['mul','matA','matC'],['mul','matB','matC']]],
                             [[['mat','m','n','matA'],['mat','m','n','matB'],['mat','n','k','matC']], ['sub',['mul','matA','matC'],['mul','matB','matC']]],
                             [[['mat','m','m','matA'],['mat','m','m','matB']], ['add',['mul','matA','matA'],['mul','matA','matB'],['mul','matB','matA'],['mul','matB','matB']]],
                             [[['mat','m','m','matA'],['mat','m','m','matB'],['def',2,'scalA']], ['add',['mul','matA','matA'],['mul','scalA',['mul','matA','matB']],['mul','matB','matB']]],
                             [[['mat','p','m','matA'],['mat','m','n','matB'],['mat','n','k','matC'],['mat','k',1,'vecD']], ['mul',['mul',['mul','matA','matB'],'matC'],'vecD']]]
    def __init__(self, other=None):
        if other is not None:
            self.dim_min = copy.deepcopy(other.dim_min)
            self.dim_max = copy.deepcopy(other.dim_max)
            self.elem_min = copy.deepcopy(other.elem_min)
            self.elem_max = copy.deepcopy(other.elem_max)
            self.expression = copy.deepcopy(other.expression)
            self.definition = copy.deepcopy(other.definition)
            self.tex_symbol = copy.deepcopy(other.tex_symbol)
            self.value = copy.deepcopy(other.value)
            self.id = copy.deepcopy(other.id)
        else:
            self.dim_min = 1
            self.dim_max = 4
            self.elem_min = -3
            self.elem_max = +3
            self.expression = [0]
            self.definition = dict()
            self.tex_symbol = dict()
            self.value = 0
            self.id = random.random()
    def __eq__(self, other):
        if not isinstance(other, MatrixComputations):
            return False
        elif self.expression != other.expression:
            return False
        elif self.definition != other.definition:
            return False
        elif self.value != other.value:
            return False
        return True
    def __hash__(self):
        self.id = hash(str(self.expression) + str(self.definition) + str(self.value))
        return self.id
    def set_dimension_range(self, dmin, dmax):
        """
        The range of dimension of vector and matrix.
        """
        self.dim_min = dmin
        self.dim_max = dmax
    def set_element_range(self, emin, emax):
        """
        The range of elements of vector and matrix.
        """
        self.elem_min = emin
        self.elem_max = emax
    def _generate_definition_in_skeleton(self, _defs, _dim_words):
        self.definition = dict()
        self.tex_symbol = dict()
        _idx_scalars = 0
        _idx_vectors = 0
        _idx_matrices = 0
        for _def in _defs:
            if _def[0] == 'mat':
                self.definition[_def[3]] = sympy.Matrix([[random.randint(self.elem_min, self.elem_max) for j in range(_dim_words[_def[2]] if type(_def[2]) is str else _def[2])] for i in range(_dim_words[_def[1]] if type(_def[1]) is str else _def[1])])
                if _def[2] == 1:
                    self.tex_symbol[_def[3]] = MatrixComputations._tex_vectors[_idx_vectors]
                    _idx_vectors+=1
                else:
                    self.tex_symbol[_def[3]] = MatrixComputations._tex_matrices[_idx_matrices]
                    _idx_matrices+=1
            elif _def[0] == 'scal':
                self.definition[_def[1]] = nonone_randint(self.elem_min, self.elem_max)
                self.tex_symbol[_def[1]] = MatrixComputations._tex_scalars[_idx_scalars]
                _idx_scalars+=1
            elif _def[0] == 'def':
                if type(_def[1]) is list:
                    self.definition[_def[2]] = sympy.Matrix(_def[1])
                    if self.definition[_def[2]].shape[1] == 1:
                        self.tex_symbol[_def[2]] = MatrixComputations._tex_vectors[_idx_vectors]
                        _idx_vectors+=1
                    else:
                        self.tex_symbol[_def[2]] = MatrixComputations._tex_matrices[_idx_matrices]
                        _idx_matrices+=1
                else:
                    self.definition[_def[2]] = _def[1]
                    self.tex_symbol[_def[2]] = MatrixComputations._tex_scalars[_idx_scalars]
                    _idx_scalars+=1
    def _evaluate(self, expr):
        if type(expr) is list:
            if expr[0] == 'add':
                temp = self._evaluate(expr[1])
                for _expr in expr[2:]:
                    temp += self._evaluate(_expr)
                return temp
            elif expr[0] == 'sub':
                return self._evaluate(expr[1]) - self._evaluate(expr[2])
            elif expr[0] == 'mul':
                return self._evaluate(expr[1]) * self._evaluate(expr[2])
            else:
                raise AttributeError('unknown header')
        elif expr in self.definition:
            return self.definition[expr]
        else:
            raise AttributeError('unknown header')
    def generate(self, t):
        [_defs, self.expression] = random.choice(MatrixComputations._skeletons[t])
        _dim_words = dict()
        for _def in _defs:
            for w in _def[1:-1]:
                _dim_words[w] = random.randint(self.dim_min, self.dim_max)
        self._generate_definition_in_skeleton(_defs, _dim_words)
        self.value = self._evaluate(self.expression)
    def get_expression(self):
        return self.expression
    def get_definition(self):
        return self.definition
    def get_value(self):
        return self.value
    def _get_incorrect_value_forgot_rest(self, expr):
        if type(expr) is list:
            if expr[0] == 'add':
                temp = self._evaluate(expr[1])
                for _expr in expr[2:-1]:
                    temp += self._evaluate(_expr)
                return temp
            elif expr[0] == 'sub':
                return self._evaluate(expr[1])
            elif expr[0] == 'mul':
                return self._evaluate(expr[1])
            else:
                raise AttributeError('unknown header')
        elif expr in self.definition:
            return self.definition[expr]
        else:
            raise AttributeError('unknown header')
    def _get_incorrect_value_forgot_next_cols(self, expr):
        _value = self._evaluate(self.expression)
        if isinstance(_value, sympy.Matrix):
            while _value.shape[1] > 1:
                _value.col_del(1)
        return _value
    def _get_incorrect_value_forgot_next_rows(self, expr):
        _value = self._evaluate(self.expression)
        if isinstance(_value, sympy.Matrix):
            while _value.shape[0] > 1:
                _value.row_del(1)
        return _value
    def _get_incorrect_value_computation_miss(self, expr):
        _value = self._evaluate(self.expression)
        if isinstance(_value, sympy.Matrix):
            _value += sympy.Matrix([[random.choice([-1,0,1]) for j in range(_value.shape[1])] for i in range(_value.shape[0])])
        else:
            _value += random.choice([-1,1])
        return _value
    def _get_incorrect_value_mul_instead_dot(self, expr):
        if type(expr) is list:
            if expr[0] == 'add':
                temp = self._get_incorrect_value_mul_instead_dot(expr[1])
                for _expr in expr[2:]:
                    temp += self._get_incorrect_value_mul_instead_dot(_expr)
                return temp
            elif expr[0] == 'sub':
                return self._get_incorrect_value_mul_instead_dot(expr[1]) - self._get_incorrect_value_mul_instead_dot(expr[2])
            elif expr[0] == 'mul':
                _matA = self._get_incorrect_value_mul_instead_dot(expr[1])
                _matB = self._get_incorrect_value_mul_instead_dot(expr[2])
                if isinstance(_matA, sympy.Matrix) and isinstance(_matB, sympy.Matrix):
                    if _matA.shape == _matB.shape:
                        _matA = _matA.tolist()
                        _matB = _matB.tolist()
                        return sympy.Matrix([[_matA[i][j]*_matB[i][j] for j in range(len(_matA[0]))] for i in range(len(_matA))])
                return _matA * _matB 
            else:
                raise AttributeError('unknown header')
        elif expr in self.definition:
            return self.definition[expr]
        else:
            raise AttributeError('unknown header')
    def _get_incorrect_value_mixed(self, expr):
        incorrect_type = random.choice(MatrixComputations._incorrect_types)
        if incorrect_type == 'forgot_rest':
            _value = self._get_incorrect_value_forgot_rest(self.expression)
        elif incorrect_type == 'forgot_next_cols':
            _value = self._get_incorrect_value_forgot_next_cols(self.expression)            
        elif incorrect_type == 'forgot_next_rows':
            _value = self._get_incorrect_value_forgot_next_rows(self.expression)         
        elif incorrect_type == 'mul_instead_dot':
            _value = self._get_incorrect_value_mul_instead_dot(self.expression)             
        else:
            _value = self._get_incorrect_value_computation_miss(self.expression)    
        if isinstance(_value, sympy.Matrix):
            _value += sympy.Matrix([[random.choice([-1,0,1]) for j in range(_value.shape[1])] for i in range(_value.shape[0])])
        else:
            _value += random.choice([-1,1])
        return _value
    def get_incorrect_value(self, incorrect_type=None):
        if incorrect_type is None:
            incorrect_type = random.choice(MatrixComputations._incorrect_types)
        if incorrect_type == 'forgot_rest':
            return self._get_incorrect_value_forgot_rest(self.expression)
        elif incorrect_type == 'forgot_next_cols':
            return self._get_incorrect_value_forgot_next_cols(self.expression)            
        elif incorrect_type == 'forgot_next_rows':
            return self._get_incorrect_value_forgot_next_rows(self.expression)            
        elif incorrect_type == 'computation_miss':
            return self._get_incorrect_value_computation_miss(self.expression)            
        elif incorrect_type == 'mul_instead_dot':
            return self._get_incorrect_value_mul_instead_dot(self.expression)              
        elif incorrect_type == 'mixed':
            return self._get_incorrect_value_mixed(self.expression)            
        else:
            raise AttributeError('unknown incorrect type')
    def _str_expression(self, expr, is_symbolic=True):
        if type(expr) is not list:
            if is_symbolic:
                return self.tex_symbol[expr]
            else:
                return sympy.latex(self.definition[expr], mat_delim='', mat_str='pmatrix')
        elif expr[0] == 'mul':
            _text = self._str_expression(expr[1], is_symbolic=is_symbolic) + r' '
            if type(expr[2]) is list:
                if expr[2][0] != 'mul':
                    _text += r'\left('
            _text += self._str_expression(expr[2], is_symbolic=is_symbolic) + r' '
            if type(expr[2]) is list:
                if expr[2][0] != 'mul':
                    _text += r'\right)'
            return _text            
        elif expr[0] == 'add':
            _text = ''
            for _expr in expr[1:]:
                _text += self._str_expression(_expr, is_symbolic=is_symbolic) + r'+'
            return _text[:-1]
        elif expr[0] == 'sub':
            _text = self._str_expression(expr[1], is_symbolic=is_symbolic) + r'-'
            if type(expr[2]) is list:
                if expr[2][0] != 'mul':
                    _text += r'\left('
            _text += self._str_expression(expr[2], is_symbolic=is_symbolic) + r' '
            if type(expr[2]) is list:
                if expr[2][0] != 'mul':
                    _text += r'\right)'
            return _text
        else:
            raise AttributeError('unknown header')
    def str_expression(self, is_latex_closure=True, is_symbolic=True):
        if is_latex_closure:
            _text = r'\( '
        else:
            _text = ''
        _text += self._str_expression(self.expression, is_symbolic=is_symbolic)
        _text = _text.replace(r'+-',r'-').replace(r'-+',r'-').replace(r'--',r'+').replace(r'++',r'+')
        _text = _text.replace(r'+1 \b',r'+ \b').replace(r'-1 \b',r'- \b')
        if is_latex_closure:
            _text += r' \)'
        return _text       
    def str_definition(self, is_latex_closure=True):
        if is_latex_closure:
            _text = r'\( '
        else:
            _text = ''
        for _key,_val in self.definition.items():
            _text += self.tex_symbol[_key] + '=' + sympy.latex(_val, mat_delim='', mat_str='pmatrix') + r',\;'
        _text = _text[:-3]
        if is_latex_closure:
            _text += r' \)'
        return _text       


# In[274]:


if __name__ == "__main__":
    mc = MatrixComputations()
    mc.set_dimension_range(1,4)
    mc.set_element_range(-3,3)
    mc.generate("msvm_22")
    display([mc.get_expression(), mc.get_definition()])
    IPython.display.display(IPython.display.HTML(mc.str_expression(is_symbolic=False)))
    IPython.display.display(IPython.display.HTML(mc.str_expression()))
    IPython.display.display(IPython.display.HTML(mc.str_definition()))
    display([mc.get_value(), mc._get_incorrect_value_mixed(mc.get_expression())])


# ## matrix in reduced row echelon form

# In[41]:


class MatrixInReducedRowEchelonForm():
    """
    This generates a general expression of reduced row echelon form.
    """
    def __init__(self, other=None):
        if other is not None:
            self.dim_min = copy.deepcopy(other.dim_min)
            self.dim_max = copy.deepcopy(other.dim_max)
            self.elem_min = copy.deepcopy(other.elem_min)
            self.elem_max = copy.deepcopy(other.elem_max)
            self.zero_vector_ratio = copy.deepcopy(other.zero_vector_ratio)
            self.zero_elements_rest = copy.deepcopy(other.zero_elements_rest)
            self.rows = copy.deepcopy(other.rows)
            self.cols = copy.deepcopy(other.cols)
            self.pivots = copy.deepcopy(other.pivots)
            self.row_space_basis = copy.deepcopy(other.row_space_basis)
            self.rref = copy.deepcopy(other.rref)
            self.matrix = copy.deepcopy(other.matrix)
            self.id = copy.deepcopy(other.id)
        else:
            self.dim_min = 3
            self.dim_max = 5
            self.elem_min = -3
            self.elem_max = +3
            self.zero_vector_ratio = 0.5
            self.zero_elements_rest = -1
            self.id = random.random()
    def __eq__(self, other):
        if not isinstance(other, MatrixReductions):
            return False
        elif self.row_space_basis != other.row_space_basis:
            return False
        elif self.rref != other.rref:
            return False
        elif self.matrix != other.matrix:
            return False
        return True
    def __hash__(self):
        self.id = hash(str(self.row_space_basis) + str(self.rref) + str(self.matrix))
        return self.id
    def set_dimension_range(self, dmin, dmax):
        """
        The range of dimension of matrix.
        """
        self.dim_min = dmin
        self.dim_max = dmax
    def set_element_range(self, emin, emax):
        """
        The range of elements of matrix.
        """
        self.elem_min = emin
        self.elem_max = emax
    def set_zero_vector_ratio(self, zratio):
        """
        The ratio of zero row vector.
        """
        self.zero_vector_ratio = zratio
    def set_zero_elements_rest(self, zratio):
        """
        The maximum number of zero elements.
        """
        self.zero_elements_rest = zratio
    def _generate_row_space_basis(self):
        _indices = [_p[1] for _p in self.pivots]
        _basis = []
        for i in range(len(self.pivots)):
            _vector = []
            for j in range(self.cols):
                if j < self.pivots[i][1]:
                    _vector.append(0)
                elif j == self.pivots[i][1]:
                    _vector.append(1)
                elif j in _indices:
                    _vector.append(0)
                else:
                    _vector.append(random.randint(self.elem_min, self.elem_max))
            _basis.append(_vector)
        return _basis
    def _generate_mixed_matrix(self):
        _op_types = ['n->kn', 'n<->m', 'n->n+km']
        _matrix = sympy.Matrix(self.row_space_basis + [random.choice(self.row_space_basis) for i in range(self.rows - len(self.row_space_basis))])
        _min = min(self.rows, self.cols) if self.zero_elements_rest < 0 else self.zero_elements_rest
        while max(abs(_matrix)) <= 2*max(abs(self.elem_min),abs(self.elem_max)) and flatten_list_all([_v[:-1] for _v in _matrix.tolist()]).count(0) > _min:
            for i in range(max([self.rows, self.cols])):
                _op = random.choice(_op_types)
                _k = 1 if random.random() < 0.5 else -1
                _row = random.choice(range(self.rows))
                _row2 = _row
                while _row == _row2:
                    _row2 = random.choice(range(self.rows))
                _matrix = _matrix.elementary_row_op(op=_op, row=_row, k=_k, row2=_row2)
        if _matrix.rank() != len(self.row_space_basis):
            return self._generate_mixed_matrix()
        for _v in _matrix.tolist():
            if is_zero_vector(_v):
                return self._generate_mixed_matrix()
        if max(abs(self.elem_min), abs(self.elem_max)) > 1:
            _list = _matrix.tolist()
            while True:
                _is_continue = False
                for i in range(len(_list)):
                    if _list[i] in _list[:i] + _list[i+1:]:
                        _k = nonzero_randint(self.elem_min, self.elem_max)
                        _matrix = _matrix.elementary_row_op(op='n->kn', row=i, k=_k)
                        _list = _matrix.tolist() 
                        _is_continue = True
                if not _is_continue:
                        break
        return _matrix.tolist()
    def generate(self, is_size_fixed=False):
        if not is_size_fixed:
            self.rows = random.randint(self.dim_min, self.dim_max)
            self.cols = max(random.randint(self.dim_min, self.dim_max), self.rows)
        self.pivots = [(0,0)]
        for i in range(1,self.cols):
            if random.random() > self.zero_vector_ratio:
                self.pivots.append((len(self.pivots),i))
        self.pivots = self.pivots[:min([len(self.pivots), self.rows, self.cols])]
        self.row_space_basis = self._generate_row_space_basis()
        self.rref = self.row_space_basis + [[0 for j in range(self.cols)] for i in range(self.rows - len(self.pivots))]
        self.matrix = self._generate_mixed_matrix()
    def get_pivots(self):
        return self.pivots
    def get_row_space_basis(self):
        return self.row_space_basis
    def get_rref(self):
        return self.rref
    def get_matrix(self):
        return self.matrix
    def str_rref(self, is_latex_closure=True):
        if is_latex_closure:
            _text = r'\( '
        else:
            _text = ''
        _text += sympy.latex(sympy.Matrix(self.rref), mat_delim='', mat_str='pmatrix')
        if is_latex_closure:
            _text += r' \)'
        return _text
    def str_matrix(self, is_latex_closure=True):
        if is_latex_closure:
            _text = r'\( '
        else:
            _text = ''
        _text += sympy.latex(sympy.Matrix(self.matrix), mat_delim='', mat_str='pmatrix')
        if is_latex_closure:
            _text += r' \)'
        return _text


# In[39]:


if __name__ == "__main__":
    mr = MatrixInReducedRowEchelonForm()
    mr.set_dimension_range(3,5)
    mr.set_element_range(-3,3)
    mr.generate()
    display(mr.rows)
    display(mr.cols)
    display(mr.pivots)
    display(mr.row_space_basis)
    display(mr.rref)
    display(mr.matrix)
    IPython.display.display(IPython.display.HTML(mr.str_rref()))
    IPython.display.display(IPython.display.HTML(mr.str_matrix()))


# ## linear equation

# In[63]:


class LinearEquation():
    """
    This generates a linear equation.
    """
    def __init__(self, other=None):
        if other is not None:
            self.dim_min = copy.deepcopy(other.dim_min)
            self.dim_max = copy.deepcopy(other.dim_max)
            self.elem_min = copy.deepcopy(other.elem_min)
            self.elem_max = copy.deepcopy(other.elem_max)
            self.zero_vector_ratio = copy.deepcopy(other.zero_vector_ratio)
            self.zero_elements_rest = copy.deepcopy(other.zero_elements_rest)
            self.inconsistent_ratio = copy.deepcopy(other.inconsistent_ratio)
            self.matrix_rref = copy.deepcopy(other.matrix_rref)
            self.solution_basis = copy.deepcopy(other.solution_basis)            
            self.id = copy.deepcopy(other.id)
        else:
            self.dim_min = 3
            self.dim_max = 5
            self.elem_min = -3
            self.elem_max = +3
            self.zero_vector_ratio = 0.5
            self.zero_elements_rest = -1
            self.inconsistent_ratio = 0.25
            self.id = random.random()
    def __eq__(self, other):
        if not isinstance(other, LinearEquation):
            return False
        elif self.matrix_rref != other.matrix_rref:
            return False
        return True
    def __hash__(self):
        self.id = hash(self.matrix_rref)
        return self.id
    def set_dimension_range(self, dmin, dmax):
        """
        The range of dimension of augmented matrix.
        """
        self.dim_min = dmin
        self.dim_max = dmax
    def set_element_range(self, emin, emax):
        """
        The range of elements of matrix.
        """
        self.elem_min = emin
        self.elem_max = emax
    def set_zero_vector_ratio(self, zratio):
        """
        The ratio of zero row vector.
        """
        self.zero_vector_ratio = zratio
    def set_zero_elements_rest(self, zratio):
        """
        The maximum number of zero elements.
        """
        self.zero_elements_rest = zratio
    def set_inconsistent_ratio(self, iratio):
        """
        The ratio of inconsistent.
        """
        self.inconsistent_ratio = iratio
    def generate_solution_basis(self, force_to_generate=False):
        if self.get_degrees_of_freedom() < 0 and not force_to_generate:
            self.solution_basis = []
            return            
        _pivots = self.matrix_rref.get_pivots()
        _const = [_v[-1] for _v in self.matrix_rref.get_row_space_basis()]
        _rsb = [_v[:-1] for _v in self.matrix_rref.get_row_space_basis()]
        if self.get_degrees_of_freedom() < 0:
            _pivots = _pivots[:-1]
            _const = _const[:-1]
            _rsb = _rsb[:-1]
        _n = len(_rsb[0])
        _zero_vector = [0 for i in range(_n)]
        _matrix = []
        _vector = []
        _current_i = 0
        for i in range(_n):
            if _current_i >= len(_pivots):
                _matrix.append(_zero_vector)
                _vector.append(0)
            elif _pivots[_current_i][1] == i:
                _matrix.append(_rsb[_current_i])
                _vector.append(_const[_current_i])
                _current_i = _current_i + 1
            else:
                _matrix.append(_zero_vector)
                _vector.append(0)
        self.solution_basis = [_vector]
        _matrix = (sympy.eye(_n) - sympy.Matrix(_matrix)).transpose().tolist()
        for _v in _matrix:
            if not is_zero_vector(_v):
                self.solution_basis.append(_v)
    def generate(self):
        self.matrix_rref = MatrixInReducedRowEchelonForm()
        self.matrix_rref.set_dimension_range(self.dim_min,self.dim_max)
        self.matrix_rref.set_element_range(self.elem_min,self.elem_max)
        self.matrix_rref.set_zero_vector_ratio(self.zero_vector_ratio)
        self.matrix_rref.set_zero_elements_rest(self.zero_elements_rest)
        self.matrix_rref.set_zero_elements_rest(1)
        if random.random() <= self.inconsistent_ratio:
            _is_inconsistent = True
        else:
            _is_inconsistent = False
        while True:
            self.matrix_rref.generate()
            _is_zero = False
            for _v in self.matrix_rref.get_matrix():
                if is_zero_vector(_v[:-1]):
                    _is_zero = True
            if _is_zero:
                continue
            for _v in (sympy.Matrix(self.matrix_rref.get_matrix()).transpose().tolist())[:-1]:
                if is_zero_vector(_v):
                    _is_zero = True
            if _is_zero:
                continue            
            if _is_inconsistent:
                if self.get_degrees_of_freedom() < 0:
                    break
            else:
                if self.get_degrees_of_freedom() >= 0:
                    break                 
        self.generate_solution_basis()
    def get_degrees_of_freedom(self):
        _m = sympy.Matrix(self.matrix_rref.get_rref())
        for _p in self.matrix_rref.get_pivots():
            if _p[1] == _m.cols - 1:
                return -1
        _num_of_variables = _m.cols - 1
        _num_of_restrictions = _m.rank()
        return _num_of_variables - _num_of_restrictions
    def get_rref(self):
        return self.matrix_rref.get_rref()
    def get_matrix(self):
        return self.matrix_rref.get_matrix()
    def get_solution_basis(self):
        return self.solution_basis
    def get_fake_solution_basis_basic_sign(self):
        _pivots = self.matrix_rref.get_pivots()
        _const = [_v[-1] for _v in self.matrix_rref.get_row_space_basis()]
        _rsb = [_v[:-1] for _v in self.matrix_rref.get_row_space_basis()]
        if self.get_degrees_of_freedom() < 0:
            _pivots = _pivots[:-1]
            _const = _const[:-1]
            _rsb = _rsb[:-1]
        _n = len(_rsb[0])
        _zero_vector = [0 for i in range(_n)]
        _matrix = []
        _vector = []
        _current_i = 0
        for i in range(_n):
            if _current_i >= len(_pivots):
                _matrix.append(_zero_vector)
                _vector.append(0)
            elif _pivots[_current_i][1] == i:
                _matrix.append(_rsb[_current_i])
                _vector.append(_const[_current_i])
                _current_i = _current_i + 1
            else:
                _matrix.append(_zero_vector)
                _vector.append(0)
        _fake_solution_basis = [_vector]
        _matrix = [[_matrix[i][j] if i == j else -_matrix[i][j] for j in range(len(_matrix[0]))] for i in range(len(_matrix))]        
        _matrix = (sympy.eye(_n) - sympy.Matrix(_matrix)).transpose().tolist()
        for _v in _matrix:
            if not is_zero_vector(_v):
                _fake_solution_basis.append(_v)
        if len(_fake_solution_basis) > 1:
            return [_fake_solution_basis]
        else:
            return []
    def get_fake_solution_basis_special_sign(self):
        _pivots = self.matrix_rref.get_pivots()
        _const = [_v[-1] for _v in self.matrix_rref.get_row_space_basis()]
        _rsb = [_v[:-1] for _v in self.matrix_rref.get_row_space_basis()]
        if self.get_degrees_of_freedom() < 0:
            _pivots = _pivots[:-1]
            _const = _const[:-1]
            _rsb = _rsb[:-1]
        _n = len(_rsb[0])
        _zero_vector = [0 for i in range(_n)]
        _matrix = []
        _vector = []
        _current_i = 0
        for i in range(_n):
            if _current_i >= len(_pivots):
                _matrix.append(_zero_vector)
                _vector.append(0)
            elif _pivots[_current_i][1] == i:
                _matrix.append(_rsb[_current_i])
                _vector.append(-_const[_current_i])
                _current_i = _current_i + 1
            else:
                _matrix.append(_zero_vector)
                _vector.append(0)
        _fake_solution_basis = [_vector]
        _matrix = (sympy.eye(_n) - sympy.Matrix(_matrix)).transpose().tolist()
        for _v in _matrix:
            if not is_zero_vector(_v):
                _fake_solution_basis.append(_v)
        if self.get_degrees_of_freedom() < 0 or not self.is_solution(_fake_solution_basis[0]):
            return [_fake_solution_basis]
        else:
            return []
    def get_fake_solution_basis_basic_drop(self):
        _pivots = self.matrix_rref.get_pivots()
        _const = [_v[-1] for _v in self.matrix_rref.get_row_space_basis()]
        _rsb = [_v[:-1] for _v in self.matrix_rref.get_row_space_basis()]
        if self.get_degrees_of_freedom() < 0:
            _pivots = _pivots[:-1]
            _const = _const[:-1]
            _rsb = _rsb[:-1]
        _n = len(_rsb[0])
        _zero_vector = [0 for i in range(_n)]
        _matrix = []
        _vector = []
        _current_i = 0
        for i in range(_n):
            if _current_i >= len(_pivots):
                _matrix.append(_zero_vector)
                _vector.append(0)
            elif _pivots[_current_i][1] == i:
                _matrix.append(_rsb[_current_i])
                _vector.append(_const[_current_i])
                _current_i = _current_i + 1
            else:
                _matrix.append(_zero_vector)
                _vector.append(0)
        _fake_solution_basis = [_vector]
        _matrix = (sympy.eye(_n) - sympy.Matrix(_matrix)).transpose().tolist()
        for _v in _matrix:
            if not is_zero_vector(_v):
                _fake_solution_basis.append(_v)
        if len(_fake_solution_basis) > 1:
            _fakes = []
            for i in range(1,len(_fake_solution_basis)):
                _fakes.append(_fake_solution_basis[:i] + _fake_solution_basis[i+1:])
            return _fakes
        else:
            return []
    def get_fake_solution_basis_basic_fulldrop(self):
        _pivots = self.matrix_rref.get_pivots()
        _const = [_v[-1] for _v in self.matrix_rref.get_row_space_basis()]
        _rsb = [_v[:-1] for _v in self.matrix_rref.get_row_space_basis()]
        if self.get_degrees_of_freedom() < 0:
            _pivots = _pivots[:-1]
            _const = _const[:-1]
            _rsb = _rsb[:-1]
        _n = len(_rsb[0])
        _zero_vector = [0 for i in range(_n)]
        _matrix = []
        _vector = []
        _current_i = 0
        for i in range(_n):
            if _current_i >= len(_pivots):
                _matrix.append(_zero_vector)
                _vector.append(0)
            elif _pivots[_current_i][1] == i:
                _matrix.append(_rsb[_current_i])
                _vector.append(_const[_current_i])
                _current_i = _current_i + 1
            else:
                _matrix.append(_zero_vector)
                _vector.append(0)
        _fake_solution_basis = [_vector]
        _matrix = (sympy.eye(_n) - sympy.Matrix(_matrix)).transpose().tolist()
        for _v in _matrix:
            if not is_zero_vector(_v):
                return [_fake_solution_basis]
        return []
    def get_fake_solution_basis_special_random(self):
        _pivots = self.matrix_rref.get_pivots()
        _const = [_v[-1] for _v in self.matrix_rref.get_row_space_basis()]
        _rsb = [_v[:-1] for _v in self.matrix_rref.get_row_space_basis()]
        _n = len(_rsb[0])
        if len(_const) > _n + 1:
            _const = _const[:_n+1]
        elif len(_const) < _n + 1:
            _const = _const + [0 for i in range(_n + 1 - len(_const))]
        _fakes = []
        for _v in [_const[:i] + _const[i+1:] for i in range(_n+1)]:
            if not self.is_solution(_v):
                _fakes.append([_v])
        return _fakes            
    def get_fake_solution_basis_special_drop(self):
        _pivots = self.matrix_rref.get_pivots()
        _const = [_v[-1] for _v in self.matrix_rref.get_row_space_basis()]
        _rsb = [_v[:-1] for _v in self.matrix_rref.get_row_space_basis()]
        if self.get_degrees_of_freedom() < 0:
            _pivots = _pivots[:-1]
            _const = _const[:-1]
            _rsb = _rsb[:-1]
        _n = len(_rsb[0])
        _zero_vector = [0 for i in range(_n)]
        _matrix = []
        _vector = []
        _current_i = 0
        for i in range(_n):
            if _current_i >= len(_pivots):
                _matrix.append(_zero_vector)
                _vector.append(0)
            elif _pivots[_current_i][1] == i:
                _matrix.append(_rsb[_current_i])
                _vector.append(0)
                _current_i = _current_i + 1
            else:
                _matrix.append(_zero_vector)
                _vector.append(0)
        _fake_solution_basis = [_vector]
        _matrix = (sympy.eye(_n) - sympy.Matrix(_matrix)).transpose().tolist()
        for _v in _matrix:
            if not is_zero_vector(_v):
                _fake_solution_basis.append(_v)
        if self.get_degrees_of_freedom() < 0 or not self.is_solution(_fake_solution_basis[0]):
            return [_fake_solution_basis]
        else:
            return []
    def get_fake_solution_basis(self, itype):
        if itype == 'basic_fulldrop':
            return self.get_fake_solution_basis_basic_fulldrop()
        elif itype == 'basic_drop':
            return self.get_fake_solution_basis_basic_drop()
        elif itype == 'special_drop':
            return self.get_fake_solution_basis_special_drop()
        elif itype == 'basic_sign':
            return self.get_fake_solution_basis_basic_sign()
        elif itype == 'special_sign':
            return self.get_fake_solution_basis_special_sign()
        elif itype == 'special_random':
            return self.get_fake_solution_basis_special_random()
        else:
            return []
    def is_solution(self, vect):
        _matAT = sympy.Matrix(self.get_matrix())
        return _matAT[:,:-1]*sympy.Matrix(vect) == _matAT[:,-1]
    def str_equation(self, is_latex_closure=True):
        if is_latex_closure:
            _text = r'\( '
        else:
            _text = ''
        _text += r'\left\{\begin{array}{l}'
        _matrix = self.get_matrix()
        _vars = [sympy.Symbol('x' + str(i)) for i in range(1,len(_matrix[0]))]
        for _v in _matrix:
            _eq = 0
            for _i in range(len(_v[:-1])):
                _eq += _v[_i] * _vars[_i]
            _text += sympy.latex(sympy.Eq(_eq, _v[-1])) + r'\\'
        _text += r'\end{array}\right.'
        if is_latex_closure:
            _text += r' \)'
        return _text            
    def str_solution(self, is_latex_closure=True):
        if len(self.solution_basis) == 0:
            return r'解なし'
        if is_latex_closure:
            _text = r'\( '
        else:
            _text = ''
        _vars = [sympy.Symbol('x' + str(i)) for i in range(1,len(self.get_matrix()[0]))]
        _text += sympy.latex(sympy.Matrix([_vars]).transpose(), mat_delim='', mat_str='pmatrix')
        _text += r' = '
        for i in range(1,len(self.solution_basis)):
            _text += sympy.latex(sympy.Matrix([self.solution_basis[i]]).transpose(), mat_delim='', mat_str='pmatrix')
            _text += r' c_{' + str(i) + r'} + '
        if is_zero_vector(self.solution_basis[0]) and len(self.solution_basis) > 1:
            _text = _text[:-2]
        else:
            _text += sympy.latex(sympy.Matrix([self.solution_basis[0]]).transpose(), mat_delim='', mat_str='pmatrix')
        for i in range(1,len(self.solution_basis)):
            _text += r',\;c_{' + str(i) + r'}\in\mathbb{R}'
        if is_latex_closure:
            _text += r' \)'
        return _text            
    def str_rref(self, is_latex_closure=True):
        if is_latex_closure:
            _text = r'\( '
        else:
            _text = ''
        _text += sympy.latex(sympy.Matrix(self.get_rref()), mat_delim='', mat_str='pmatrix')
        if is_latex_closure:
            _text += r' \)'
        return _text
    def str_matrix(self, is_latex_closure=True):
        if is_latex_closure:
            _text = r'\( '
        else:
            _text = ''
        _text += sympy.latex(sympy.Matrix(self.get_matrix()), mat_delim='', mat_str='pmatrix')
        if is_latex_closure:
            _text += r' \)'
        return _text


# In[71]:


if __name__ == "__main__":
    le = LinearEquation()
    le.set_dimension_range(3,4)
    le.set_element_range(-3,3)
    le.generate()
    display(le.get_degrees_of_freedom())
    IPython.display.display(IPython.display.HTML(le.str_rref()))
    IPython.display.display(IPython.display.HTML(le.str_matrix()))
    IPython.display.display(IPython.display.HTML(le.str_equation()))
    IPython.display.display(IPython.display.HTML(le.str_solution()))
    le.generate_solution_basis(force_to_generate=True)
    IPython.display.display(IPython.display.HTML(le.str_solution()))
    display(le.get_fake_solution_basis('basic_sign'))
    display(le.get_fake_solution_basis('special_sign'))
    display(le.get_fake_solution_basis('basic_fulldrop'))
    display(le.get_fake_solution_basis('basic_drop'))
    display(le.get_fake_solution_basis('special_drop'))
    display(le.get_fake_solution_basis('special_random'))


# ## matrix in row echelon form

# In[13]:


class MatrixInRowEchelonForm():
    """
    This generates a general expression of row echelon form.
    """
    def __init__(self, other=None):
        if other is not None:
            self.dim_min = copy.deepcopy(other.dim_min)
            self.dim_max = copy.deepcopy(other.dim_max)
            self.elem_min = copy.deepcopy(other.elem_min)
            self.elem_max = copy.deepcopy(other.elem_max)
            self.zero_vector_ratio = copy.deepcopy(other.zero_vector_ratio)
            self.rows = copy.deepcopy(other.rows)
            self.cols = copy.deepcopy(other.cols)
            self.pivots = copy.deepcopy(other.pivots)
            self.row_space_basis = copy.deepcopy(other.row_space_basis)
            self.ref = copy.deepcopy(other.ref)
            self.matrix = copy.deepcopy(other.matrix)
            self.id = copy.deepcopy(other.id)
        else:
            self.dim_min = 3
            self.dim_max = 5
            self.elem_min = -3
            self.elem_max = +3
            self.zero_vector_ratio = 0.5
            self.id = random.random()
    def __eq__(self, other):
        if not isinstance(other, MatrixReductions):
            return False
        elif self.row_space_basis != other.row_space_basis:
            return False
        elif self.ref != other.ref:
            return False
        elif self.matrix != other.matrix:
            return False
        return True
    def __hash__(self):
        self.id = hash(str(self.row_space_basis) + str(self.ref) + str(self.matrix))
        return self.id
    def set_dimension_range(self, dmin, dmax):
        """
        The range of dimension of matrix.
        """
        self.dim_min = dmin
        self.dim_max = dmax
    def set_element_range(self, emin, emax):
        """
        The range of elements of matrix.
        """
        self.elem_min = emin
        self.elem_max = emax
    def set_zero_vector_ratio(self, zratio):
        """
        The ratio of zero row vector.
        """
        self.zero_vector_ratio = zratio
    def _generate_row_space_basis(self):
        _indices = [_p[1] for _p in self.pivots]
        _basis = []
        for i in range(len(self.pivots)):
            _vector = []
            for j in range(self.cols):
                if j < self.pivots[i][1]:
                    _vector.append(0)
                elif j == self.pivots[i][1]:
                    _vector.append(nonzero_randint(self.elem_min, self.elem_max))
                else:
                    _vector.append(random.randint(self.elem_min, self.elem_max))
            _basis.append(_vector)
        return _basis
    def _generate_mixed_matrix(self):
        _op_types = ['n->kn', 'n<->m', 'n->n+km']
        _matrix = sympy.Matrix(self.row_space_basis + [random.choice(self.row_space_basis) for i in range(self.rows - len(self.row_space_basis))])
        while max(abs(_matrix)) <= 4*max(abs(self.elem_min),abs(self.elem_max)) and flatten_list_all(_matrix.tolist()).count(0) > min(self.rows, self.cols):
            for i in range(max([self.rows, self.cols])):
                _op = random.choice(_op_types)
                _k = 1 if random.random() < 0.5 else -1
                _row = random.choice(range(self.rows))
                _row2 = _row
                while _row == _row2:
                    _row2 = random.choice(range(self.rows))
                _matrix = _matrix.elementary_row_op(op=_op, row=_row, k=_k, row2=_row2)
        if _matrix.rank() != len(self.row_space_basis):
            return self._generate_mixed_matrix()
        return _matrix.tolist()
    def _generate_swapped_matrix(self):
        _op_types = ['n<->m']
        _matrix = sympy.Matrix(self.row_space_basis + [[0 for j in range(self.cols)] for i in range(self.rows - len(self.row_space_basis))])
        for i in range(max([self.rows, self.cols])):
            _op = random.choice(_op_types)
            _k = 1 if random.random() < 0.5 else -1
            _row = random.choice(range(self.rows))
            _row2 = _row
            while _row == _row2:
                _row2 = random.choice(range(self.rows))
            _matrix = _matrix.elementary_row_op(op=_op, row=_row, k=_k, row2=_row2)
        if _matrix.rank() != len(self.row_space_basis):
            return self._generate_swapped_matrix()
        return _matrix.tolist()
    def generate(self, is_size_fixed=False, is_swap_only=False):
        if not is_size_fixed:
            self.rows = random.randint(self.dim_min, self.dim_max)
            self.cols = max(random.randint(self.dim_min, self.dim_max), self.rows)
        self.pivots = [(0,0)]
        for i in range(1,self.cols):
            if random.random() > self.zero_vector_ratio:
                self.pivots.append((len(self.pivots),i))
        self.pivots = self.pivots[:min([len(self.pivots), self.rows, self.cols])]
        self.row_space_basis = self._generate_row_space_basis()
        self.ref = self.row_space_basis + [[0 for j in range(self.cols)] for i in range(self.rows - len(self.pivots))]
        if is_swap_only:
            self.matrix = self._generate_swapped_matrix()
        else:
            self.matrix = self._generate_mixed_matrix()
    def get_rank(self):
        return len(self.pivots)
    def get_ref(self):
        return self.ref
    def get_matrix(self):
        return self.matrix
    def str_ref(self, is_latex_closure=True):
        if is_latex_closure:
            _text = r'\( '
        else:
            _text = ''
        _text += sympy.latex(sympy.Matrix(self.ref), mat_delim='', mat_str='pmatrix')
        if is_latex_closure:
            _text += r' \)'
        return _text
    def str_matrix(self, is_latex_closure=True):
        if is_latex_closure:
            _text = r'\( '
        else:
            _text = ''
        _text += sympy.latex(sympy.Matrix(self.matrix), mat_delim='', mat_str='pmatrix')
        if is_latex_closure:
            _text += r' \)'
        return _text


# In[178]:


if __name__ == "__main__":
    mr = MatrixInRowEchelonForm()
    mr.set_dimension_range(3,5)
    mr.set_element_range(-3,3)
    mr.generate()
    display(mr.rows)
    display(mr.cols)
    display(mr.pivots)
    display(mr.row_space_basis)
    display(mr.ref)
    display(mr.matrix)
    IPython.display.display(IPython.display.HTML(mr.str_ref()))
    IPython.display.display(IPython.display.HTML(mr.str_matrix()))


# ## inverse matrix

# In[13]:


class InverseMatrix():
    """
    This generates a general expression of matrix will be inversed.
    """
    def __init__(self, other=None):
        if other is not None:
            self.dim_min = copy.deepcopy(other.dim_min)
            self.dim_max = copy.deepcopy(other.dim_max)
            self.square_ratio = copy.deepcopy(other.square_ratio)
            self.singular_ratio = copy.deepcopy(other.singular_ratio)
            self.num_swap = copy.deepcopy(other.num_swap)
            self.num_add = copy.deepcopy(other.num_add)
            self.num_scale = copy.deepcopy(other.num_scale)
            self.scale_min = copy.deepcopy(other.scale_min)
            self.scale_max = copy.deepcopy(other.scale_max)
            self.rows = copy.deepcopy(other.rows)
            self.cols = copy.deepcopy(other.cols)
            self.row_space_generator = copy.deepcopy(other.row_space_generator)
            self.matrix = copy.deepcopy(other.matrix)
            self.is_singular = copy.deepcopy(other.is_singular)
            self.matrix_extended = copy.deepcopy(other.matrix_extended)
            self.matrix_extended_rref = copy.deepcopy(other.matrix_extended_rref)
            self.inverse_matrix = copy.deepcopy(other.inverse_matrix)
            self.id = copy.deepcopy(other.id)
        else:
            self.dim_min = 3
            self.dim_max = 5
            self.square_ratio = 0.75
            self.singular_ratio = 0.5
            self.num_swap = 2
            self.num_add = 10
            self.num_scale = 0            
            self.scale_min = -2
            self.scale_max = +2
            self.id = random.random()
    def __eq__(self, other):
        if not isinstance(other, MatrixReductions):
            return False
        elif self.row_space_generator != other.row_space_generator:
            return False
        elif self.matrix != other.matrix:
            return False
        elif self.inverse_matrix != other.inverse_matrix:
            return False
        return True
    def __hash__(self):
        self.id = hash(str(self.row_space_generator) + str(self.matrix) + str(self.inverse_matrix))
        return self.id
    def set_dimension_range(self, dmin, dmax):
        """
        The range of dimension of matrix.
        """
        self.dim_min = dmin
        self.dim_max = dmax
    def set_ratio(self, square_ratio, singular_ratio):
        """
        The ratios of generated matrix.
        """
        self.square_ratio = square_ratio
        self.singular_ratio = singular_ratio
    def set_elementary_operation(self, num_swap, num_add, num_scale, scale_min, scale_max):
        """
        The configurations of elementary row operations.
        """
        self.num_swap = num_swap
        self.num_add = num_add
        self.num_scale = num_scale
        self.scale_min = scale_min
        self.scale_max = scale_max
    def _generate_row_space_generator(self):
        self.rows = random.randint(self.dim_min, self.dim_max)
        self.cols = self.rows
        if random.random() > self.square_ratio:
            while self.rows == self.cols:
                self.cols = random.randint(self.dim_min, self.dim_max)
        if self.cols == self.rows and random.random() > self.singular_ratio:
            self.is_singular = False
            return [[0 if i != j else 1 for j in range(self.cols)] for i in range(self.rows)]
        else:
            self.is_singular = True
        _maxrows = self.rows
        while _maxrows >= self.cols:
            _maxrows -= 1
        _generator = []
        while sympy.Matrix(_generator).rank() == 0:
            _generator = [[random.choice([-1,0,1]) for j in range(self.cols)] for i in range(_maxrows)]
        while len(_generator) < self.rows:
            _generator.append(random.choice(_generator))
        return _generator
    def _generate_matrix(self):
        _matrix = sympy.Matrix(self.row_space_generator)
        for i in range(self.num_add):
            _k = random.choice([-1,1])
            _row = random.choice(range(self.rows))
            _row2 = _row
            while _row == _row2:
                _row2 = random.choice(range(self.rows))
            _matrix = _matrix.elementary_row_op(op='n->n+km', row=_row, k=_k, row2=_row2)
        for i in range(self.num_swap):
            _row = random.choice(range(self.rows))
            _row2 = _row
            while _row == _row2:
                _row2 = random.choice(range(self.rows))
            _matrix = _matrix.elementary_row_op(op='n<->m', row=_row, row2=_row2)
        for i in range(self.num_scale):
            _k = nonzero_randint(self.scale_min, self.scale_max)
            _row = random.choice(range(self.rows))
            _matrix = _matrix.elementary_row_op(op='n->kn', row=_row, k=_k)
        return _matrix.tolist()
    def _generate_extended_matrix(self):
        _identity = [[0 if i != j else 1 for j in range(self.rows)] for i in range(self.rows)]
        _matrix = [_l + _r for _l,_r in zip(self.matrix, _identity)]
        return _matrix
    def _generate_extended_matrix_rref(self):
        _rref = sympy.Matrix(self.matrix_extended).rref()[0]
        return _rref.tolist()
    def _generate_inverse_matrix(self):
        if not self.is_singular:
            _inv = sympy.Matrix(self.matrix)**-1
            return _inv.tolist()
        return [_elems[self.cols:] for _elems in self.matrix_extended_rref]
    def generate(self):
        self.row_space_generator = self._generate_row_space_generator()
        self.matrix = self._generate_matrix()
        self.matrix_extended = self._generate_extended_matrix()
        self.matrix_extended_rref = self._generate_extended_matrix_rref()
        self.inverse_matrix = self._generate_inverse_matrix()    
    def get_inverse_matrix(self):
        return self.inverse_matrix
    def str_inverse_matrix(self, is_latex_closure=True):
        if is_latex_closure:
            _text = r'\( '
        else:
            _text = ''
        _text += sympy.latex(sympy.Matrix(self.inverse_matrix), mat_delim='', mat_str='pmatrix')
        if is_latex_closure:
            _text += r' \)'
        return _text
    def get_matrix(self):
        return self.matrix
    def str_matrix(self, is_latex_closure=True):
        if is_latex_closure:
            _text = r'\( '
        else:
            _text = ''
        _text += sympy.latex(sympy.Matrix(self.matrix), mat_delim='', mat_str='pmatrix')
        if is_latex_closure:
            _text += r' \)'
        return _text
    def get_matrix_extended(self):
        return self.matrix_extended
    def str_matrix_extended(self, is_latex_closure=True):
        if is_latex_closure:
            _text = r'\( '
        else:
            _text = ''
        _text += sympy.latex(sympy.Matrix(self.matrix_extended), mat_delim='', mat_str='pmatrix')
        if is_latex_closure:
            _text += r' \)'
        return _text
    def get_matrix_extended_rref(self):
        return self.matrix_extended_rref
    def str_matrix_extended_rref(self, is_latex_closure=True):
        if is_latex_closure:
            _text = r'\( '
        else:
            _text = ''
        _text += sympy.latex(sympy.Matrix(self.matrix_extended_rref), mat_delim='', mat_str='pmatrix')
        if is_latex_closure:
            _text += r' \)'
        return _text


# In[39]:


if __name__ == "__main__":
    mr = InverseMatrix()
    mr.set_dimension_range(3,5)
    mr.set_ratio(0.25, 0.5)
    mr.set_elementary_operation(2, 10, 0, -2, 2)
    mr.generate()
    display(mr.rows)
    display(mr.cols)
    display(mr.row_space_generator)
    display(mr.is_singular)
    IPython.display.display(IPython.display.HTML(mr.str_matrix()))
    IPython.display.display(IPython.display.HTML(mr.str_inverse_matrix()))
    IPython.display.display(IPython.display.HTML(mr.str_matrix_extended()))
    IPython.display.display(IPython.display.HTML(mr.str_matrix_extended_rref()))


# In[ ]:




