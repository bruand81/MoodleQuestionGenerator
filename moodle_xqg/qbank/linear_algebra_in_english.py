#!/usr/bin/env python
# coding: utf-8

# Moodle XML Question Generator, (C) 2020 Kosaku Nagasaka (Kobe University)

# note
# > ***This notebook includes only some English ported generators (only 8 of more than 100).***

# ## generate the module file

# In[ ]:


if __name__ == "__main__":
    import subprocess
    subprocess.run(['jupyter', 'nbconvert', '--to', 'python', '_linear_algebra_in_english.ipynb','--output','linear_algebra_in_english.py'])


# # Linear Algebra

# In[1]:


if __name__ == "__main__":
    from inspect import getsourcefile
    import os.path
    import sys
    current_path = os.path.abspath(getsourcefile(lambda:0))
    current_dir = os.path.dirname(current_path)
    parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
    sys.path.insert(0, parent_dir)
    import core
    import common
    import linear_algebra
    import linear_algebra_next_generation
else:
    from .. import core
    from . import common
    from . import linear_algebra
    from . import linear_algebra_next_generation


# In[2]:


import sympy
import sympy.combinatorics
import random
import IPython
import itertools
import functools
import copy


# ## NOTES

# - Any matrix will be translated in the latex format with "pmatrix". Please translate it with your preferable environment (e.g. bmatrix) by the option of the top level function.
# - Any vector symbol will be translated in the latex format with "\vec". Please translate it with your preferable command (e.g. \boldsymbol) by the option of the top level function.

# ## elementary row operation for result (en)

# In[3]:


class elementary_row_operation_for_result(core.Question):
    name = 'elementary row operation to get the desired matrix'
    op_types = ['n->kn', 'n<->m', 'n->n+km']
    def __init__(self, dmin=2, dmax=4, emin=-3, emax=3):
        # range of matrix size
        self.dim_min = dmin
        self.dim_max = dmax
        # range of elements
        self.elem_min = emin
        self.elem_max = emax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='elementary row operation', quiz_number=_quiz_number)
        _rows = random.randint(self.dim_min,self.dim_max)
        _cols = random.randint(self.dim_min,self.dim_max)
        _matA = [[random.randint(self.elem_min,self.elem_max) for i in range(_cols)] for j in range(_rows)]
        _op_type = random.choice(self.op_types)
        _k = random.randint(self.elem_min,self.elem_max)
        _row = random.choice(range(_rows))
        _row2 = random.choice(range(_rows))
        while _row2 == _row:
            _row2 = random.choice(range(_rows))
        _matB = sympy.Matrix(_matA).elementary_row_op(op=_op_type, row=_row, k=_k, row2=_row2).tolist()
        quiz.quiz_identifier = hash(str(_matA)+str(_op_type)+str(_matB))
        # generate the answer
        quiz.data = [_rows, _cols, _matA, _op_type, _k, _row, _row2, _matB]
        ans = { 'fraction': 100, 'data': [_op_type, _k, _row, _row2] }
        quiz.answers.append(ans)
        return quiz      
    def correct_answers_generate(self, quiz, size=1):
        # nothing to do since we already have the correct answer.
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_rows, _cols, _matA, _op_type, _k, _row, _row2, _matB] = quiz.data
        for i in range(_rows):
            for j in range(_rows):
                for t in self.op_types:
                    if t != 'n->kn' and i == j:
                        continue
                    if t == 'n->kn':
                        j = 0
                    incorrectB = sympy.Matrix(_matA).elementary_row_op(op=t, row=i, k=_k, row2=j)
                    if incorrectB != sympy.Matrix(_matB):
                        ans['feedback'] = r'this operation does not produce the matrix \( B \), and this fact can be confirmed by applying the operation to the matrix \( A \).'
                        ans['data'] = [t, _k, i, j]
                        answers.append(dict(ans))
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_rows, _cols, _matA, _op_type, _k, _row, _row2, _matB] = quiz.data
        _text = r'Which elementary row operation is required to get the following matrix \( B \) from the following matrix \( A \).<br />'
        _text += r'\( A=' + sympy.latex(sympy.Matrix(_matA), mat_delim='', mat_str='pmatrix')
        _text += r',\;B=' + sympy.latex(sympy.Matrix(_matB), mat_delim='', mat_str='pmatrix') + r' \)'
        return _text
    def _num2str(self, num):
        if num == 1:
            return '1st'
        elif num == 2:
            return '2nd'
        elif num == 3:
            return '3rd'
        else:
            return str(num)+'th'
    def answer_text(self, ans):
        [_op_type, _k, _row, _row2] = ans['data']
        if _op_type == 'n->kn':
            return r'multiply the ' + self._num2str(_row+1) + r' row by ' + str(_k) + r'.'
        elif _op_type == 'n<->m':
            return r'interchange the ' + self._num2str(_row+1) + r' and ' + self._num2str(_row2+1) + r' rows.'
        else: # 'n->n+km'
            return r'add the ' + self._num2str(_row2+1) + r' row multiplied by ' + str(_k) + r' to the '  + self._num2str(_row+1) + r' row.'


# In[4]:


if __name__ == "__main__":
    q = elementary_row_operation_for_result()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[5]:


if __name__ == "__main__":
    pass
    #qz.save('elementary_row_operation_for_result_in_english.xml')


# ## inverse matrix with reduced row eachelon form (en)

# In[3]:


class inverse_matrix_with_rref(core.Question):
    name = 'inverse computation with rref of concatenation with identity matrix'
    def __init__(self, dmin=2, dmax=4, sqrate=0.75, singrate=0.5, nswap=2, nadd=10, nsca=0, smin=-2, smax=2):
        # range of matrix size
        self.dim_min = dmin
        self.dim_max = dmax
        # probability of square matrix
        self.square_ratio = sqrate
        # probability of singular matrix if square
        self.singular_ratio = singrate
        # number of row elementary operations for matrix generation
        self.num_swap = nswap
        self.num_add = nadd
        self.num_scale = nsca
        # range of scalar (for row elementary operation)
        self.scale_min = smin
        self.scale_max = smax        
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='inverse computation with rref', quiz_number=_quiz_number)
        _mr = linear_algebra_next_generation.InverseMatrix()
        _mr.set_dimension_range(self.dim_min, self.dim_max)
        _mr.set_ratio(self.square_ratio, self.singular_ratio)
        _mr.set_elementary_operation(self.num_swap, self.num_add, self.num_scale, self.scale_min, self.scale_max)
        _mr.generate()
        quiz.quiz_identifier = hash(_mr)
        quiz.data = [_mr]
        # generate the answer
        if _mr.is_singular:
            if _mr.rows != _mr.cols:
                _ans = r'This is not a square matrix.'
            else:
                _ans = r'The rank is not equal to the number of rows hence this is singular.'
        else:
            _ans = _mr.get_inverse_matrix()
        ans = { 'fraction': 100, 'data': _ans }
        quiz.answers.append(ans)
        return quiz      
    def correct_answers_generate(self, quiz, size=1):
        # nothing to do since we already have the correct answer.
        pass
    def incorrect_answers_generate(self, quiz, size=5):
        answers = []
        ans = { 'fraction': 0 }
        [_mr] = quiz.data
        if _mr.is_singular:     
            if _mr.rows != _mr.cols:
                ans['feedback'] = r'It is a necessary condition for a matrix to be invertible that it is a square matrix.'
                ans['data'] = _mr.get_inverse_matrix()
                answers.append(dict(ans))
                ans['data'] = [[-_elem for _elem in _row] for _row in ans['data']]
                answers.append(dict(ans))                
                ans['data'] = [_row[-_mr.cols:] for _row in _mr.get_matrix_extended_rref()]
                answers.append(dict(ans))
                ans['data'] = [[-_elem for _elem in _row] for _row in ans['data']]
                answers.append(dict(ans))                
            else:
                ans['feedback'] = r'Check the definition of a square matrix. This matrix is square.'
                ans['data'] = r'This is not a square matrix.'
                answers.append(dict(ans))
                ans['feedback'] = r'Check the definition of the rank. It is not invertible if the rank is different from the number of rows.'
                ans['data'] = _mr.get_inverse_matrix()
                answers.append(dict(ans))
                ans['data'] = [[-_elem for _elem in _row] for _row in ans['data']]
                answers.append(dict(ans))                
        else:     
            ans['feedback'] = r'Check the definition of a square matrix. This matrix is square.'
            ans['data'] = r'This is not a square matrix.'
            answers.append(dict(ans)) 
            ans['feedback'] = r'Check the definition of the rank. It is invertible if the rank is equal to the number of rows.'
            ans['data'] = r'The rank is not equal to the number of rows hence this is singular.'
            answers.append(dict(ans)) 
            ans['feedback'] = r'The right part of the RREF is the inverse matrix as is.'
            ans['data'] = [[-_elem for _elem in _row] for _row in _mr.get_inverse_matrix()]
            if sympy.Matrix(_mr.get_inverse_matrix()) != sympy.Matrix(ans['data']):
                answers.append(dict(ans))     
        ans['feedback'] = r'Check the definition of the inverse matrix. Changing the sign of the matrix does not make the inverse in general.'           
        ans['data'] = [[-_elem for _elem in _row] for _row in _mr.get_matrix()]
        if sympy.Matrix(_mr.get_inverse_matrix()) != sympy.Matrix(ans['data']):
            answers.append(dict(ans))         
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_mr] = quiz.data
        _text = r'Choose the inverse matrix of the following matrix \( A \).<br />\( A = '
        _text += _mr.str_matrix(is_latex_closure=False)
        _text += r' \)<br />For your information, the matrix on the right is the reduced row echelon form of the matrix on the left.<br />'
        _text += r'\( ' + sympy.latex(sympy.Matrix(_mr.get_matrix_extended()), mat_delim='', mat_str='pmatrix') 
        _text += r'\;\rightarrow\;' + sympy.latex(sympy.Matrix(_mr.get_matrix_extended_rref()), mat_delim='', mat_str='pmatrix') + r' \)'
        return _text
    def answer_text(self, ans):
        if isinstance(ans['data'], str):
            return ans['data']
        else:
            return r'\( ' + sympy.latex(sympy.Matrix(ans['data']), mat_delim='', mat_str='pmatrix') + r' \)'


# In[4]:


if __name__ == "__main__":
    q = inverse_matrix_with_rref(dmin=2, dmax=2, sqrate=1, nswap=1, nadd=10, nsca=1)
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=25) 


# In[5]:


if __name__ == "__main__":
    pass
    #qz.save('inverse_matrix_with_rref_small_in_english.xml')


# In[6]:


if __name__ == "__main__":
    core._force_to_this_lang = 'ja'
    q = inverse_matrix_with_rref()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=25)


# In[9]:


if __name__ == "__main__":
    pass
    #qz.save('inverse_matrix_with_rref_in_english.xml')


# ## minor expansion (en)

# In[9]:


class minor_expansion(core.Question):
    name = 'determinant expansion by minors'
    def __init__(self, dmin=3, dmax=4, emin=-3, emax=3):
        # range of matrix size
        self.dim_min = dmin
        self.dim_max = dmax
        # range of elements
        self.elem_min = emin
        self.elem_max = emax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='determinant expansion by minors', quiz_number=_quiz_number)
        _dim = random.randint(self.dim_min, self.dim_max)
        _matrix = [[random.randint(self.elem_min, self.elem_max) for _j in range(_dim)] for _i in range(_dim)]
        _is_row = True if random.random() < 0.5 else False
        _rowcol = random.choice(range(_dim))
        quiz.quiz_identifier = hash(str(_dim) + str(_matrix) + str(_is_row) + str(_rowcol))
        # generate the answer
        _ans = self._make_expansion(_is_row, _rowcol, _matrix)
        ans = { 'fraction': 100, 'data': _ans }
        quiz.answers.append(ans)
        quiz.data = [_dim, _matrix, _is_row, _rowcol, _ans]
        return quiz
    def _make_expansion(self, _is_row, _rowcol, _matrix, _base=-1, _padding=0, _rowcolM=None):
        _ans = []
        _dim = len(_matrix)
        if _rowcolM is None:
            _rowcolM = _rowcol
        if _is_row:
            for _c in range(_dim):
                _submatrix = sympy.Matrix(_matrix)
                _submatrix.row_del(_rowcolM)
                _submatrix.col_del(_c)
                _ans.append([_matrix[_rowcol][_c]*(_base)**(_rowcol+_c+_padding),_submatrix.tolist()])
        else:
            for _r in range(_dim):
                _submatrix = sympy.Matrix(_matrix)
                _submatrix.row_del(_r)
                _submatrix.col_del(_rowcolM)
                _ans.append([_matrix[_r][_rowcol]*(_base)**(_r+_rowcol+_padding),_submatrix.tolist()])
        return _ans        
    def correct_answers_generate(self, quiz, size=1):
        # nothing to do
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_dim, _matrix, _is_row, _rowcol, _ans] = quiz.data
        _incorrect_rowcol = random.choice(list(set(range(_dim))-set([_rowcol])))
        _incorrect_ans = self._make_expansion(_is_row, _rowcol, _matrix, _rowcolM=_incorrect_rowcol)
        if _incorrect_ans != _ans:
            ans['feedback'] = r'Take care of the location of the specified row or column.'
            ans['data'] = _incorrect_ans
            answers.append(dict(ans))
        _incorrect_ans = self._make_expansion(not _is_row, _rowcol, _matrix, _rowcolM=_incorrect_rowcol)
        if _incorrect_ans != _ans:
            ans['feedback'] = r'Take care of the location of the specified row or column, and do not mix up row and column.'
            ans['data'] = _incorrect_ans
            answers.append(dict(ans))
        if _dim > 2:
            _incorrect_rowcol = random.choice(list(set(range(_dim))-set([_rowcol,_incorrect_rowcol])))
            _incorrect_ans = self._make_expansion(_is_row, _rowcol, _matrix, _rowcolM=_incorrect_rowcol)
            if _incorrect_ans != _ans:
                ans['feedback'] = r'Take care of the location of the specified row or column.'
                ans['data'] = _incorrect_ans
                answers.append(dict(ans))
            _incorrect_ans = self._make_expansion(not _is_row, _rowcol, _matrix, _rowcolM=_incorrect_rowcol)
            if _incorrect_ans != _ans:
                ans['feedback'] = r'Take care of the location of the specified row or column, and do not mix up row and column.'
                ans['data'] = _incorrect_ans
                answers.append(dict(ans))
        _incorrect_rowcol = random.choice(list(set(range(_dim))-set([_rowcol])))
        _incorrect_ans = self._make_expansion(_is_row, _incorrect_rowcol, _matrix)
        if _incorrect_ans != _ans:
            ans['feedback'] = r'Take care of the location of the specified row or column.'
            ans['data'] = _incorrect_ans
            answers.append(dict(ans))
        _incorrect_ans = self._make_expansion(not _is_row, _incorrect_rowcol, _matrix)
        if _incorrect_ans != _ans:
            ans['feedback'] = r'Take care of the location of the specified row or column, and do not mix up row and column.'
            ans['data'] = _incorrect_ans
            answers.append(dict(ans))
        _incorrect_ans = self._make_expansion(not _is_row, _rowcol, _matrix)
        if _incorrect_ans != _ans:
            ans['feedback'] = r'Do not mix up row and column.'
            ans['data'] = _incorrect_ans
            answers.append(dict(ans))
        _incorrect_ans = self._make_expansion(_is_row, _rowcol, _matrix, _base=1)
        if _incorrect_ans != _ans:
            ans['feedback'] = r'The signs of summands are changing. Take care of signs.'
            ans['data'] = _incorrect_ans
            answers.append(dict(ans))            
        _incorrect_ans = self._make_expansion(not _is_row, _rowcol, _matrix, _base=1)
        if _incorrect_ans != _ans:
            ans['feedback'] = r'The signs of summands are changing. Take care of signs, and do not mix up row and column.'
            ans['data'] = _incorrect_ans
            answers.append(dict(ans))
        _incorrect_ans = self._make_expansion(_is_row, _rowcol, _matrix, _padding=1)
        if _incorrect_ans != _ans:
            ans['feedback'] = r'The signs are incorrect. Take care of signs.'
            ans['data'] = _incorrect_ans
            answers.append(dict(ans))            
        _incorrect_ans = self._make_expansion(not _is_row, _rowcol, _matrix, _padding=1)
        if _incorrect_ans != _ans:
            ans['feedback'] = r'The signs are incorrect. Take care of signs, and do not mix up row and column.'
            ans['data'] = _incorrect_ans
            answers.append(dict(ans))            
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_dim, _matrix, _is_row, _rowcol, _ans] = quiz.data
        _text = r'Choose the (Laplace/cofactor) expansion of the following matrix along the ' + str(_rowcol+1)
        if _rowcol == 0:
            _text += r'st '
        elif _rowcol == 1:
            _text += r'nd '
        elif _rowcol == 2:
            _text += r'rd '
        else:
            _text += r'th '
        if _is_row:
            _text += r'row'
        else:
            _text += r'column'
        _text += r'.<br />\( '
        _text += sympy.latex(sympy.Matrix(_matrix), mat_delim='', mat_str='vmatrix') 
        return _text + r' \)'
    def answer_text(self, ans):
        _text = r'\( '
        for _me in ans['data']:
            if _me[0] < 0:
                _text += sympy.latex(_me[0])
            else:
                _text += r'+' + sympy.latex(_me[0])
            _text += r'\times'
            _text += sympy.latex(sympy.Matrix(_me[1]), mat_delim='', mat_str='vmatrix') 
        _text += r' \)'
        return _text


# In[10]:


if __name__ == "__main__":
    q = minor_expansion()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=25)


# In[11]:


if __name__ == "__main__":
    pass
    #qz.save('minor_expansion_in_english.xml')


# ## adjugate matrix with minors (en)

# In[3]:


class adjugate_matrix_with_minors(core.Question):
    name = 'adjugate matrix (with the minors computed)'
    def __init__(self, dmin=3, dmax=4, emin=-3, emax=3):
        # range of matrix size
        self.dim_min = dmin
        self.dim_max = dmax
        # range of elements
        self.elem_min = emin
        self.elem_max = emax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='adjugate matrix', quiz_number=_quiz_number)
        _dim = random.randint(self.dim_min, self.dim_max)
        _matrix = [[random.randint(self.elem_min, self.elem_max) for _j in range(_dim)] for _i in range(_dim)]
        _adjugate = self._make_adjugate(_matrix)
        quiz.quiz_identifier = hash(str(_dim) + str(_matrix) + str(_adjugate))
        # generate the answer
        ans = { 'fraction': 100, 'data': _adjugate }
        quiz.answers.append(ans)
        quiz.data = [_dim, _matrix, _adjugate]
        return quiz
    def _make_adjugate(self, _matrix):
        _dim = len(_matrix)
        _adjugate = [[sympy.Matrix(self._make_minor(_matrix,_j,_i)).det()*(-1)**(_i+_j) for _j in range(_dim)] for _i in range(_dim)]
        return _adjugate
    def _make_minor(self, _matrix, _i, _j):
        _minor = sympy.Matrix(_matrix)
        _minor.row_del(_i)
        _minor.col_del(_j)
        return _minor.tolist()
    def correct_answers_generate(self, quiz, size=1):
        # nothing to do
        pass
    def incorrect_answers_generate(self, quiz, size=5):
        answers = []
        ans = { 'fraction': 0 }
        [_dim, _matrix, _adjugate] = quiz.data
        ans['feedback'] = r'be careful on signs of elements of adjugate matrix.'
        _incorrect_ans = [[sympy.Matrix(self._make_minor(_matrix,_j,_i)).det()*(1)**(_i+_j) for _j in range(_dim)] for _i in range(_dim)]
        if _incorrect_ans != _adjugate:
            ans['data'] = _incorrect_ans
            answers.append(dict(ans))
        _incorrect_ans = [[sympy.Matrix(self._make_minor(_matrix,_j,_i)).det()*(-1)**(_i+_j+1) for _j in range(_dim)] for _i in range(_dim)]
        if _incorrect_ans != _adjugate:
            ans['data'] = _incorrect_ans
            answers.append(dict(ans))
        ans['feedback'] = r'be careful on signs and subscript indices of elements of adjugate matrix.'
        _incorrect_ans = [[sympy.Matrix(self._make_minor(_matrix,_i,_j)).det()*(1)**(_i+_j) for _j in range(_dim)] for _i in range(_dim)]
        if _incorrect_ans != _adjugate:
            ans['data'] = _incorrect_ans
            answers.append(dict(ans))
        _incorrect_ans = [[sympy.Matrix(self._make_minor(_matrix,_i,_j)).det()*(-1)**(_i+_j+1) for _j in range(_dim)] for _i in range(_dim)]
        if _incorrect_ans != _adjugate:
            ans['data'] = _incorrect_ans
            answers.append(dict(ans))
        ans['feedback'] = r'be careful on subscript indices of elements of adjugate matrix.'
        _incorrect_ans = [[sympy.Matrix(self._make_minor(_matrix,_i,_j)).det()*(-1)**(_i+_j) for _j in range(_dim)] for _i in range(_dim)]
        if _incorrect_ans != _adjugate:
            ans['data'] = _incorrect_ans
            answers.append(dict(ans))
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_dim, _matrix, _adjugate] = quiz.data
        _text = r'Choose the adjugate matrix of the following \( A \).<br />\( A='
        _text += sympy.latex(sympy.Matrix(_matrix), mat_delim='', mat_str='pmatrix') + r' \)<br />'
        _text += r'For your information, we have the minors:<br />\( '
        for _i in range(_dim):
            for _j in range(_dim):
                _text += r'|A_{' + str(_i+1) + r',' + str(_j+1) + r'}|=' + sympy.latex(sympy.Matrix(self._make_minor(_matrix,_i,_j)).det()) + r',\;'
        _text = _text[:-3]
        _text += r' \)'
        return _text
    def answer_text(self, ans):
        _text = r'\( ' + sympy.latex(sympy.Matrix(ans['data']), mat_delim='', mat_str='pmatrix') + r' \)'
        return _text


# In[4]:


if __name__ == "__main__":
    q = adjugate_matrix_with_minors()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=25)


# In[8]:


if __name__ == "__main__":
    pass
    #qz.save('adjugate_matrix_with_minors_in_english.xml')


# ## linearly independent check equation with basis (en)

# In[22]:


class linearly_independent_check_equation_with_basis(core.Question):
    name = 'linear equation for linearly independent check (with basis)'
    def __init__(self, emin=-3, emax=3, dmin=2, dmax=3, rdefimin=0, rdefimax=1, rdunmin=0, rdunmax=1):
        # range of elements
        self.emin = emin
        self.emax = emax
        # range of vector space dimensions
        self.dmin = dmin
        self.dmax = dmax
        # range of rank deficiency from the dimension of the vector space
        self.rdefimin = rdefimin
        self.rdefimax = rdefimax
        # range of number of redundant generators
        self.rdunmin = rdunmin
        self.rdunmax = rdunmax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='linear equation for linearly independent check', quiz_number=_quiz_number)
        ls = linear_algebra.LinearSpace()
        ls.set_dimension_range(self.dmin, self.dmax)
        ls.set_element_range(self.emin, self.emax)
        ls.set_rank_deficient_range(self.rdefimin, self.rdefimax)
        ls.set_redundant_range(self.rdunmin, self.rdunmax)
        ls.set_keep_integer(True)
        ls.generate()
        while ls.num_gens < 2:
            ls.generate()
        quiz.data = ls
        quiz.quiz_identifier = hash(ls)
        # generate the answer
        ans = { 'fraction': 100, 'data': self._str_linear_eq(sympy.Matrix(random.sample(ls.generator, len(ls.generator))).transpose()) }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # nothing to do since we already have the correct answer.
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        ls = quiz.data
        _correct_matrix = sympy.Matrix(random.sample(ls.generator, len(ls.generator))).transpose()
        if _correct_matrix != _correct_matrix.transpose():
            ans['feedback'] = 'each vector is multiplied by the uknowns (variables). collecting the coefficients gets the equation.'
            ans['data'] = self._str_linear_eq(_correct_matrix.transpose())
            answers.append(dict(ans))
        # generates incorrect choices randomly (simple random generation is not recommended though)
        count = 0
        while len(answers) < size and count < 100:
            count += 1
            ls_sub = linear_algebra.LinearSpace(ls)
            _matrix = random.sample(ls_sub.generator, len(ls_sub.generator))
            r = random.randint(0, ls_sub.num_gens - 1)
            c = random.randint(0, ls_sub.dimension - 1)
            _matrix[r][c] += random.choice([-1,1])
            if random.random() < 0.5:
                ans['feedback'] = 'each vector is multiplied by the uknowns (variables). collecting the coefficients gets the equation.'
                ans['data'] = self._str_linear_eq(sympy.Matrix(_matrix)) 
            else:
                ans['feedback'] = 'the order of vectors is not important for linearly independent check.'
                ans['data'] = self._str_linear_eq(sympy.Matrix(_matrix).transpose()) 
            answers.append(dict(ans))
            answers = common.answer_union(answers)
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        _text = 'Suppose that ' + self._str_us(quiz.data) + ' are linearly independent. '
        _text += 'Choose the linear equation that is suitable to check whether ' + self._str_vs(quiz.data)
        _text += ' are linearly independent or not.<br />'
        _text += self._str_vectors_as_u(quiz.data)
        return _text    
    def answer_text(self, ans):
        return ans['data']
    def _str_linear_eq(self, matrix):
        _text = r'\( ' + sympy.latex(matrix, mat_delim='', mat_str='pmatrix') + r'\begin{pmatrix}'
        for i in range(matrix.cols):
            _text += r'c_{' + str(i+1) + r'}'
            if i < matrix.cols - 1:
                _text += r'\\ '
        _text += r'\end{pmatrix}=\vec{0}' + r' \)'
        return _text
    def _str_us(self, ls):
        _text = r'\( '
        for i in range(ls.dimension):
            _text += r'\vec{u}_{' + str(i+1) + r'}'
            if i < ls.dimension - 1:
                _text += r',\;'        
        return _text + r' \)'
    def _str_vs(self, ls):
        _text = r'\( '
        for i in range(ls.num_gens):
            _text += r'\vec{v}_{' + str(i+1) + r'}'
            if i < ls.num_gens - 1:
                _text += r',\;'        
        return _text + r' \)'
    def _str_vectors_as_u(self, ls):
        _text = r'\( '
        for i in range(ls.num_gens):
            _text += r'\vec{v}_{' + str(i+1) + r'}='
            _is_first = True
            for j in range(ls.dimension):
                if ls.generator[i][j] == 1:
                    if _is_first:
                        _text += r'\vec{u}_{' + str(j+1) + r'}'
                        _is_first = False
                    else:
                        _text += r'+\vec{u}_{' + str(j+1) + r'}'
                elif ls.generator[i][j] == -1:
                    _is_first = False
                    _text += r'-\vec{u}_{' + str(j+1) + r'}'
                elif ls.generator[i][j] > 1:
                    if _is_first:
                        _text += str(ls.generator[i][j]) + r'\vec{u}_{' + str(j+1) + r'}'
                        _is_first = False
                    else:
                        _text += r'+' + str(ls.generator[i][j]) + r'\vec{u}_{' + str(j+1) + r'}'
                elif ls.generator[i][j] < -1:
                    _is_first = False
                    _text += str(ls.generator[i][j]) + r'\vec{u}_{' + str(j+1) + r'}'
            if i < ls.num_gens - 1:
                _text += r',\;'        
        return _text + r' \)'


# In[23]:


if __name__ == "__main__":
    q = linearly_independent_check_equation_with_basis()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=12)


# In[24]:


if __name__ == "__main__":
    pass
    #qz.save('linearly_independent_check_equation_with_basis_in_english.xml')


# ## reduced row echelon form and linearly dependent relation (en)

# In[29]:


class rref_and_linearly_dependent_relation(core.Question):
    name = 'linearly dependent relation of columns is elementary row operations invariant'
    def __init__(self, emin=-3, emax=3, dmin=3, dmax=5, rdefimin=1, rdefimax=3, rdunmin=0, rdunmax=0):
        # range of elements
        self.emin = emin
        self.emax = emax
        # range of vector space dimensions
        self.dmin = dmin
        self.dmax = dmax
        # range of rank deficiency from the dimension of the vector space
        self.rdefimin = rdefimin
        self.rdefimax = rdefimax
        # range of number of redundant generators
        self.rdunmin = rdunmin
        self.rdunmax = rdunmax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='linearly dependent relation by rref', quiz_number=_quiz_number)
        ls = linear_algebra.LinearSpace()
        ls.set_dimension_range(self.dmin, self.dmax)
        ls.set_element_range(self.emin, self.emax)
        ls.set_rank_deficient_range(self.rdefimin, self.rdefimax)
        ls.set_redundant_range(self.rdunmin, self.rdunmax)
        ls.set_keep_integer(True)
        ls.set_pivot_one(True)
        ls.generate()
        while ls.num_gens < 2:
            ls.generate()
        quiz.data = ls
        quiz.quiz_identifier = hash(ls)
        # generate the answer
        ans = { 'fraction': 100, 'data': self._str_rref_to_us(ls, conv_pivot=ls.pivot_positions)}
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # nothing to do since we already have the correct answer.
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        ls = quiz.data
        # sign change
        ls_sub = linear_algebra.LinearSpace(ls)
        for i in range(len(ls_sub.basis)):
            for j in range(len(ls_sub.basis[i])):
                if j not in ls_sub.pivot_positions:
                    ls_sub.basis[i][j] *= -1
        ans['feedback'] = 'linearly dependent relation of columns is not changed by any elementary row operations. be careful for signs.'
        ans['data'] = self._str_rref_to_us(ls_sub, conv_pivot=ls_sub.pivot_positions)
        answers.append(dict(ans))
        # gather pivots
        if ls.pivot_positions[-1] != len(ls.pivot_positions)-1 and ls.pivot_positions[-1] < ls.dimension - 1:
            _conv = [i for i in range(len(ls_sub.pivot_positions))]
            ans['feedback'] = 'be careful for positions of pivots that are vectors may form a basis, and also careful for signs.'
            ans['data'] = self._str_rref_to_us(ls_sub, conv_pivot=_conv)
            answers.append(dict(ans))
            ls_sub = linear_algebra.LinearSpace(ls)
            ans['feedback'] = 'be careful for positions of pivots that are vectors may form a basis.'
            ans['data'] = self._str_rref_to_us(ls_sub, conv_pivot=_conv)
            answers.append(dict(ans))
        # generates incorrect choices randomly (simple random generation is not recommended though)
        count = 0
        while len(answers) < size and count < 100:
            count += 1
            ls_sub = linear_algebra.LinearSpace(ls)
            ls_sub.generate_basis()
            if ls_sub.basis != ls.basis:
                ans['feedback'] = 'linearly dependent relation of columns is not changed by any elementary row operations. be careful for positions of pivots that are vectors may form a basis.'
                ans['data'] = self._str_rref_to_us(ls_sub, conv_pivot=ls_sub.pivot_positions)
                answers.append(dict(ans))
                answers = common.answer_union(answers)
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        _text = 'To find a maximally linearly independent subset of the following vectors ' + self._str_us(quiz.data)
        _text += ', we computed the following reduced row echelon form \( A \) of the matrix ' + self._str_us_matrix(quiz.data)
        _text += '. Take the vectors corresponding to the pivots as a basis, '
        _text += 'and choose the correct linear combinations of basis vectors for the rest of vectors.<br />'
        _text += self._str_generators_with_u(quiz.data) + r'<br />'
        _text += self._str_rref(quiz.data)
        return _text    
    def answer_text(self, ans):
        return ans['data']
    def _str_generators_with_u(self, ls):
        _text = r'\( '
        for i in range(ls.dimension):
            _text += r'\vec{u}_{' + str(i+1) + r'}='
            _text += sympy.latex(sympy.Matrix(ls.generator)[:,i], mat_delim='', mat_str='pmatrix')
            if i < ls.dimension - 1:
                _text += r',\;'
        return _text + r' \)'
    def _str_rref(self, ls):
        _text = r'\( A='
        _text += sympy.latex(sympy.Matrix(ls.basis), mat_delim='', mat_str='pmatrix')
        return _text + r' \)'
    def _str_rref_to_us(self, ls, conv_pivot):
        _text = r'\( '
        _counter = 0
        _rref_matrix = sympy.Matrix(ls.basis)
        for i in range(ls.dimension):
            if i not in ls.pivot_positions:
                _counter += 1
                _text += r'\vec{u}_{' + str(i+1) + r'}='
                _coefs = sympy.matrix2numpy(_rref_matrix[:,i].transpose()).tolist()[0]
                _is_first = True
                for j in range(len(_coefs)):
                    if _coefs[j] == 1:
                        if _is_first:
                            _text += r'\vec{u}_{' + str(conv_pivot[j]+1) + r'}'
                            _is_first = False
                        else:
                            _text += r'+\vec{u}_{' + str(conv_pivot[j]+1) + r'}'
                    elif _coefs[j] == -1:
                        _text += r'-\vec{u}_{' + str(conv_pivot[j]+1) + r'}'
                        _is_first = False
                    elif _coefs[j] > 1:
                        if _is_first:
                            _text += sympy.latex(_coefs[j]) + r'\vec{u}_{' + str(conv_pivot[j]+1) + r'}'
                            _is_first = False
                        else:
                            _text += r'+' + sympy.latex(_coefs[j]) + r'\vec{u}_{' + str(conv_pivot[j]+1) + r'}'
                    elif _coefs[j] < -1:
                        _text += sympy.latex(_coefs[j]) + r'\vec{u}_{' + str(conv_pivot[j]+1) + r'}'
                        _is_first = False                    
                if _counter < ls.dimension - len(ls.pivot_positions):
                    _text += r',\;'
        return _text + r' \)'
    def _str_us(self, ls):
        _text = r'\( '
        for i in range(ls.dimension):
            _text += r'\vec{u}_{' + str(i+1) + r'}'
            if i < ls.dimension - 1:
                _text += r',\;'
        return _text + r' \)'
    def _str_us_matrix(self, ls):
        _text = r'\( \begin{pmatrix}'
        for i in range(ls.dimension):
            _text += r'\vec{u}_{' + str(i+1) + r'}'
            if i < ls.dimension - 1:
                _text += r'\;'
        return _text + r'\end{pmatrix} \)'


# In[30]:


if __name__ == "__main__":
    q = rref_and_linearly_dependent_relation()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=2)


# In[31]:


if __name__ == "__main__":
    pass
    #qz.save('rref_and_linearly_dependent_relation_in_english.xml')


# ## select eigen vector (en)

# In[32]:


class select_eigenvector(core.Question):
    name = 'eigen vector of linear transformation'
    def __init__(self, emin=-3, emax=3, dmin=2, dmax=4, rmin=-3, rmax=3):
        # range of elements
        self.elem_min = emin
        self.elem_max = emax
        # range of vector space dimensions
        self.dim_min = dmin
        self.dim_max = dmax
        # range of eigen values
        self.root_min = rmin
        self.root_max = rmax 
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='eigen vector of linear transformation', quiz_number=_quiz_number)
        es = linear_algebra.EigenSpace()
        es.set_dimension_range(self.dim_min, self.dim_max)
        es.set_element_range(self.elem_min, self.elem_max)
        es.set_root_range(self.root_min, self.root_max)
        es.generate(is_force_diagonalizable=True, is_force_symbolic=False, is_force_numeric=False)
        quiz.quiz_identifier = hash(es)
        # generate the answer
        _idx = random.choice(range(len(es.eigen_values)))
        _mat = sympy.zeros(1,es.dimension)
        while max(abs(_mat)) == 0:
            for vec in es.eigen_vectors[_idx]:
                _mat = _mat + linear_algebra.nonzero_randint(self.elem_min, self.elem_max)*sympy.Matrix([vec])
        quiz.data = [es, _idx]
        ans = { 'fraction': 100, 'data': es.str_vector(sympy.matrix2numpy(_mat).tolist()[0], is_latex_closure=True)}
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # nothing to do since we already have the correct answer.
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        es = quiz.data[0]
        _idx = quiz.data[1]
        while len(answers) < size:
            _mat = sympy.zeros(1,es.dimension)
            _sample_idx = random.choice(range(len(es.eigen_values)))
            for vec in es.eigen_vectors[_sample_idx]:
                _mat = _mat + random.randint(self.elem_min, self.elem_max)*sympy.Matrix([vec])
            _mat = sympy.matrix2numpy(_mat).tolist()[0]
            _random_idx = random.choice(range(len(_mat)))
            _mat[_random_idx] += linear_algebra.nonzero_randint(self.elem_min, self.elem_max)            
            if not es.is_an_eigen_vector(_mat):
                ans['feedback'] = 'just transform each vector by the linear transformation, and check each image against the definition of eigen vector.'
                ans['data'] = es.str_vector(_mat, is_latex_closure=True)
                answers.append(dict(ans))                
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        es = quiz.data[0]
        vec = quiz.data[1]
        _text = r'Choose an eigen vector of the following linear transformation.<br />'        
        _text += es.str_map(is_latex_closure=True)
        return _text
    def answer_text(self, ans):
        return ans['data']


# In[34]:


if __name__ == "__main__":
    q = select_eigenvector()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=12)


# In[35]:


if __name__ == "__main__":
    pass
    #qz.save('select_eigenvector_in_english.xml')


# ## expanding inner expression with numerical values (en)

# In[3]:


class expanding_inner_expression_with_numerical_values(core.Question):
    name = 'expansion of inner or norm expression by linearity'
    def __init__(self, emin=-2, emax=2, dmin=2, dmax=3):
        # range of elements
        self.elem_min = emin
        self.elem_max = emax
        # range of number of vectors
        self.dim_min = dmin
        self.dim_max = dmax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='expansion of inner or norm expression by linearity', quiz_number=_quiz_number)
        ins = linear_algebra.InnerSpace(domain='Rn')
        _num_of_vectors = random.randint(self.dim_min, self.dim_max)
        _vecs = [[linear_algebra.nonzero_randint(self.elem_min, self.elem_max) for i in range(2)] for j in range(_num_of_vectors)]
        _inners = [[0 for i in range(_num_of_vectors)] for j in range(_num_of_vectors)]
        for i in range(_num_of_vectors):
            for j in range(_num_of_vectors):
                _inners[i][j] = ins.inner_product(_vecs[i], _vecs[j])
        _expr = ins.generate_symbolic(_num_of_vectors, self.elem_min, self.elem_max)
        quiz.quiz_identifier = hash(str(_expr))
        # generate the answer
        _value = ins.eval_symbolic(_expr, _inners)
        quiz.data = [_num_of_vectors, _inners, _expr, _value]
        ans = { 'fraction': 100, 'data': _value }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # nothing to do since we already have the correct answer.
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_num_of_vectors, _inners, _expr, _value] = quiz.data
        ins = linear_algebra.InnerSpace(domain=None)
        ans['feedback'] = '\( V \) is a normed vector space hence we can calculate the expression.'
        ans['data'] = r'it depends on the actual definition of inner product hence unknown.'
        answers.append(dict(ans))
        loop = 0
        while len(answers) < size and loop < 10:
            incorrect_value = ins.eval_symbolic(_expr, _inners, wrong=True)
            if _value != incorrect_value:
                ans['feedback'] = 'to get the value, expand the expression by the linearity and replace the innder products with their value. be careful on calculation.'
                ans['data'] = incorrect_value
                answers.append(dict(ans))
            answers = common.answer_union(answers)
            loop += 1
        while len(answers) < size:
            incorrect_value = _value + linear_algebra.nonzero_randint(-5,5)
            if _value != incorrect_value:
                ans['feedback'] = 'be careful on calculation.'
                ans['data'] = incorrect_value
                answers.append(dict(ans))
            answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_num_of_vectors, _inners, _expr, _value] = quiz.data
        ins = linear_algebra.InnerSpace(domain=None)
        _text = r'Let \( V \) be a inner product space (normed vector space) over \( \mathbb{R} \). '
        _text += r'With its inner product, we have the following values of inner products of vectors.<br />'
        for i in range(_num_of_vectors):
            for j in range(i, _num_of_vectors):
                _text += r'\(\left(\vec{v_{' + str(i+1) + r'}},\vec{v_{' + str(j+1) + r'}}\right)=' + sympy.latex(_inners[i][j]) + r'\)'
                if i < _num_of_vectors - 1 or j < _num_of_vectors - 1:
                    _text += r', '
        _text += r'<br />Then, choose the value of the following expression (innder product or norm).<br />'
        _text += ins.str_symbolic(_expr, is_latex_closure=True, variable='v')
        return _text
    def answer_text(self, ans):
        if isinstance(ans['data'], str):
            _text = ans['data']
        else:
            _text = r'\(' + sympy.latex(ans['data']) + r'\)'
        return _text


# In[6]:


if __name__ == "__main__":
    q = expanding_inner_expression_with_numerical_values()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[7]:


if __name__ == "__main__":
    pass
    #qz.save('expanding_inner_expression_with_numerical_values_in_english.xml')


# ## dummy

# In[ ]:





# # All the questions

# In[ ]:


questions_str = ['elementary_row_operation_for_result', 'inverse_matrix_with_rref',
                 'minor_expansion', 'adjugate_matrix_with_minors', 
                 'linearly_independent_check_equation_with_basis', 'rref_and_linearly_dependent_relation', 
                 'select_eigenvector', 'expanding_inner_expression_with_numerical_values']
questions = [eval(q) for q in questions_str]
def listview():
    fullhtml = '<ul>'
    for i in range(len(questions_str)):
        fullhtml += '<li>' + questions_str[i] + ':' + questions[i].name + '</li>'
    fullhtml += '</ul>'       
    IPython.display.display(IPython.display.HTML(fullhtml))
def generate():
    instances = []
    for q in questions:
        instances.append(q())
    return instances


# In[ ]:




