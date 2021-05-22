#!/usr/bin/env python
# coding: utf-8

# Moodle XML Question Generator, (C) 2020 Kosaku Nagasaka (Kobe University)

# ## モジュールファイルの生成

# In[1]:


if __name__ == "__main__":
    import subprocess
    subprocess.run(['jupyter', 'nbconvert', '--to', 'python', '_linear_algebra_2.ipynb','--output','linear_algebra_2.py'])


# # Linear Algebra 2 (inverse matrix and determinant)

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
    import linear_algebra_next_generation
    core._force_to_this_lang = 'ja'
else:
    from .. import core
    from . import common
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

# ## singular or non-singular of row echelon form

# In[7]:


class singular_or_nonsingular_ref(core.Question):
    name = '正則性の判定（階段行列の場合）'
    def __init__(self, dmin=3, dmax=4, emin=-3, emax=3, zratio=0.25):
        # 行列のサイズの範囲
        self.dim_min = dmin
        self.dim_max = dmax
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 零成分となる確率
        self.zero_ratio = zratio
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='正則判定（階段行列）', quiz_number=_quiz_number)
        _mr = linear_algebra_next_generation.MatrixInRowEchelonForm()
        _mr.set_dimension_range(self.dim_min,self.dim_max)
        _mr.set_element_range(self.elem_min,self.elem_max)
        _mr.set_zero_vector_ratio(self.zero_ratio)
        _mr.generate()
        quiz.quiz_identifier = hash(_mr)
        # 正答の選択肢の生成
        _ref = sympy.Matrix(_mr.get_ref())
        quiz.data = [_ref]
        if _ref.rows == _ref.cols:
            if _ref.rank() == _ref.rows:
                _ans = r'正則行列である。'
            else:
                _ans = r'階数が行数と異なり正則ではない。'
        else:
            _ans = r'そもそも正方行列ではない。'
        ans = { 'fraction': 100, 'data': _ans }
        quiz.answers.append(ans)
        return quiz      
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=5):
        answers = []
        ans = { 'fraction': 0 }
        [_ref] = quiz.data
        if _ref.rows == _ref.cols:
            if _ref.rank() == _ref.rows:
                ans['feedback'] = r'階数の定義を確認してください。'
                ans['data'] = r'階数が行数と異なり正則ではない。'
                answers.append(dict(ans)) 
                ans['feedback'] = r'正方行列とは，行数と列数が同じ行列のことです。'
                ans['data'] = r'そもそも正方行列ではない。'
                answers.append(dict(ans)) 
            else:
                ans['feedback'] = r'正則行列の同値な条件は，階数が行数と同じ行列のことです。'
                ans['data'] = r'正則行列である。'
                answers.append(dict(ans)) 
                ans['feedback'] = r'正方行列とは，行数と列数が同じ行列のことです。'
                ans['data'] = r'そもそも正方行列ではない。'
                answers.append(dict(ans)) 
        else:
            ans['feedback'] = r'正則行列は，少なくとも正方行列である必要があります。'
            ans['data'] = r'正則行列である。'
            answers.append(dict(ans)) 
            ans['feedback'] = r'正則行列は，少なくとも正方行列である必要があります。'
            ans['data'] = r'階数が行数と異なり正則ではない。'
            answers.append(dict(ans)) 
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_ref] = quiz.data
        _text = r'次の行列について，最も適切な言及をしているものを選択してください。<br />'
        _text += r'\( ' + sympy.latex(sympy.Matrix(_ref), mat_delim='', mat_str='pmatrix') + r' \)'
        return _text
    def answer_text(self, ans):
        return ans['data']


# In[8]:


if __name__ == "__main__":
    q = singular_or_nonsingular_ref()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=25)


# In[9]:


if __name__ == "__main__":
    pass
    #qz.save('singular_or_nonsingular_ref.xml')


# ## singular or non-singular of nearly row echelon form

# In[10]:


class singular_or_nonsingular_nearly_ref(core.Question):
    name = '正則性の判定（ほぼ階段行列の場合）'
    def __init__(self, dmin=3, dmax=4, emin=-3, emax=3, zratio=0.25):
        # 行列のサイズの範囲
        self.dim_min = dmin
        self.dim_max = dmax
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 零成分となる確率
        self.zero_ratio = zratio
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='正則判定（ほぼ階段行列）', quiz_number=_quiz_number)
        _mr = linear_algebra_next_generation.MatrixInRowEchelonForm()
        _mr.set_dimension_range(self.dim_min,self.dim_max)
        _mr.set_element_range(self.elem_min,self.elem_max)
        _mr.set_zero_vector_ratio(self.zero_ratio)
        _mr.generate(is_swap_only=True)
        quiz.quiz_identifier = hash(_mr)
        # 正答の選択肢の生成
        _ref = sympy.Matrix(_mr.get_matrix())
        quiz.data = [_ref]
        if _ref.rows == _ref.cols:
            if _ref.rank() == _ref.rows:
                _ans = r'正則行列である。'
            else:
                _ans = r'階数が行数と異なり正則ではない。'
        else:
            _ans = r'そもそも正方行列ではない。'
        ans = { 'fraction': 100, 'data': _ans }
        quiz.answers.append(ans)
        return quiz      
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=5):
        answers = []
        ans = { 'fraction': 0 }
        [_ref] = quiz.data
        if _ref.rows == _ref.cols:
            if _ref.rank() == _ref.rows:
                ans['feedback'] = r'階数の定義を確認してください。'
                ans['data'] = r'階数が行数と異なり正則ではない。'
                answers.append(dict(ans)) 
                ans['feedback'] = r'正方行列とは，行数と列数が同じ行列のことです。'
                ans['data'] = r'そもそも正方行列ではない。'
                answers.append(dict(ans)) 
            else:
                ans['feedback'] = r'正則行列の同値な条件は，階数が行数と同じ行列のことです。'
                ans['data'] = r'正則行列である。'
                answers.append(dict(ans)) 
                ans['feedback'] = r'正方行列とは，行数と列数が同じ行列のことです。'
                ans['data'] = r'そもそも正方行列ではない。'
                answers.append(dict(ans)) 
        else:
            ans['feedback'] = r'正則行列は，少なくとも正方行列である必要があります。'
            ans['data'] = r'正則行列である。'
            answers.append(dict(ans)) 
            ans['feedback'] = r'正則行列は，少なくとも正方行列である必要があります。'
            ans['data'] = r'階数が行数と異なり正則ではない。'
            answers.append(dict(ans)) 
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_ref] = quiz.data
        _text = r'次の行列について，最も適切な言及をしているものを選択してください。<br />'
        _text += r'\( ' + sympy.latex(_ref, mat_delim='', mat_str='pmatrix') + r' \)'
        return _text
    def answer_text(self, ans):
        return ans['data']


# In[11]:


if __name__ == "__main__":
    q = singular_or_nonsingular_nearly_ref()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=25)


# In[12]:


if __name__ == "__main__":
    pass
    #qz.save('singular_or_nonsingular_nearly_ref.xml')


# ## inverse matrix with reduced row eachelon form

# In[7]:


class inverse_matrix_with_rref(core.Question):
    name = '逆行列の計算（拡大した行列の掃き出し結果付き）'
    def __init__(self, dmin=2, dmax=4, sqrate=0.75, singrate=0.5, nswap=2, nadd=10, nsca=0, smin=-2, smax=2):
        # 行列のサイズの範囲
        self.dim_min = dmin
        self.dim_max = dmax
        # 正方行列となる確率
        self.square_ratio = sqrate
        # 正方のときに特異となる確率
        self.singular_ratio = singrate
        # 生成のための行の基本変形の階数
        self.num_swap = nswap
        self.num_add = nadd
        self.num_scale = nsca
        # スカラー倍の際のスカラーの範囲
        self.scale_min = smin
        self.scale_max = smax        
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='逆行列の計算（掃き出し済み）', quiz_number=_quiz_number)
        _mr = linear_algebra_next_generation.InverseMatrix()
        _mr.set_dimension_range(self.dim_min, self.dim_max)
        _mr.set_ratio(self.square_ratio, self.singular_ratio)
        _mr.set_elementary_operation(self.num_swap, self.num_add, self.num_scale, self.scale_min, self.scale_max)
        _mr.generate()
        quiz.quiz_identifier = hash(_mr)
        quiz.data = [_mr]
        # 正答の選択肢の生成
        if _mr.is_singular:
            if _mr.rows != _mr.cols:
                _ans = r'そもそも正方行列ではない。'
            else:
                _ans = r'階数が行数と異なり正則ではない。'
        else:
            _ans = _mr.get_inverse_matrix()
        ans = { 'fraction': 100, 'data': _ans }
        quiz.answers.append(ans)
        return quiz      
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=5):
        answers = []
        ans = { 'fraction': 0 }
        [_mr] = quiz.data
        if _mr.is_singular:     
            if _mr.rows != _mr.cols:
                ans['feedback'] = r'逆行列が存在するための必要条件の1つが，正方行列であることです。'
                ans['data'] = _mr.get_inverse_matrix()
                answers.append(dict(ans))
                ans['data'] = [[-_elem for _elem in _row] for _row in ans['data']]
                answers.append(dict(ans))                
                ans['data'] = [_row[-_mr.cols:] for _row in _mr.get_matrix_extended_rref()]
                answers.append(dict(ans))
                ans['data'] = [[-_elem for _elem in _row] for _row in ans['data']]
                answers.append(dict(ans))                
            else:
                ans['feedback'] = r'正方行列の定義を確認しましょう。この行列は正方行列です。'
                ans['data'] = r'そもそも正方行列ではない。'
                answers.append(dict(ans))
                ans['feedback'] = r'階数の定義を確認してください。階数が行数と一致しなければ逆行列は存在しません。'
                ans['data'] = _mr.get_inverse_matrix()
                answers.append(dict(ans))
                ans['data'] = [[-_elem for _elem in _row] for _row in ans['data']]
                answers.append(dict(ans))                
        else:     
            ans['feedback'] = r'正方行列の定義を確認しましょう。この行列は正方行列です。'
            ans['data'] = r'そもそも正方行列ではない。'
            answers.append(dict(ans)) 
            ans['feedback'] = r'階数の定義を確認してください。階数と行数は一致しています。'
            ans['data'] = r'階数が行数と異なり正則ではない。'
            answers.append(dict(ans)) 
            ans['feedback'] = r'逆行列の計算では，掃き出したままが結果となります。'
            ans['data'] = [[-_elem for _elem in _row] for _row in _mr.get_inverse_matrix()]
            if sympy.Matrix(_mr.get_inverse_matrix()) != sympy.Matrix(ans['data']):
                answers.append(dict(ans))     
        ans['feedback'] = r'逆行列の定義を確認しましょう。符号反転しただけでは一般に逆行列にはなりません。'           
        ans['data'] = [[-_elem for _elem in _row] for _row in _mr.get_matrix()]
        if sympy.Matrix(_mr.get_inverse_matrix()) != sympy.Matrix(ans['data']):
            answers.append(dict(ans))         
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_mr] = quiz.data
        _text = r'次の行列の逆行列を選択してください。<br />'
        _text += _mr.str_matrix(is_latex_closure=True)
        _text += r'<br />ただし，次の左側の行列を簡約すると，右側の行列になることを参考にしても構いません。<br />'
        _text += r'\( ' + sympy.latex(sympy.Matrix(_mr.get_matrix_extended()), mat_delim='', mat_str='pmatrix') 
        _text += r'\;\rightarrow\;' + sympy.latex(sympy.Matrix(_mr.get_matrix_extended_rref()), mat_delim='', mat_str='pmatrix') + r' \)'
        return _text
    def answer_text(self, ans):
        if isinstance(ans['data'], list):
            return r'\( ' + sympy.latex(sympy.Matrix(ans['data']), mat_delim='', mat_str='pmatrix') + r' \)'
        return ans['data']


# In[19]:


if __name__ == "__main__":
    q = inverse_matrix_with_rref(dmin=2, dmax=2, sqrate=1, nswap=1, nadd=10, nsca=1)
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=25) 


# In[20]:


if __name__ == "__main__":
    pass
    #qz.save('inverse_matrix_with_rref_small.xml')


# In[8]:


if __name__ == "__main__":
    q = inverse_matrix_with_rref()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=25)


# In[9]:


if __name__ == "__main__":
    pass
    #qz.save('inverse_matrix_with_rref.xml')


# ## inverse matrix

# In[12]:


class inverse_matrix(core.Question):
    name = '逆行列の計算（結果に含まれる要素での確認）'
    def __init__(self, dmin=2, dmax=4, sqrate=0.80, singrate=0.20, nswap=2, nadd=10, nsca=1, smin=-2, smax=2):
        # 行列のサイズの範囲
        self.dim_min = dmin
        self.dim_max = dmax
        # 正方行列となる確率
        self.square_ratio = sqrate
        # 正方のときに特異となる確率
        self.singular_ratio = singrate
        # 生成のための行の基本変形の階数
        self.num_swap = nswap
        self.num_add = nadd
        self.num_scale = nsca
        # スカラー倍の際のスカラーの範囲
        self.scale_min = smin
        self.scale_max = smax        
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='逆行列の計算（要素での確認）', quiz_number=_quiz_number)
        _mr = linear_algebra_next_generation.InverseMatrix()
        _mr.set_dimension_range(self.dim_min, self.dim_max)
        _mr.set_ratio(self.square_ratio, self.singular_ratio)
        _mr.set_elementary_operation(self.num_swap, self.num_add, self.num_scale, self.scale_min, self.scale_max)
        _mr.generate()
        quiz.quiz_identifier = hash(_mr)
        # 正答の選択肢の生成
        if _mr.is_singular:
            if _mr.rows != _mr.cols:
                _ans = r'そもそも正方行列ではない。'
            else:
                _ans = r'階数が行数と異なり正則ではない。'
        else:
            _ans = set(linear_algebra_next_generation.flatten_list_all(_mr.get_inverse_matrix()))
        ans = { 'fraction': 100, 'data': _ans }
        _inv_matrix_feedback = r'逆行列は，\( ' + sympy.latex(sympy.Matrix(_mr.get_inverse_matrix()), mat_delim='', mat_str='pmatrix')  + r' \)となります。'
        if not _mr.is_singular:
            ans['feedback'] = _inv_matrix_feedback
        quiz.answers.append(ans)
        quiz.data = [_mr, _inv_matrix_feedback, _ans]
        return quiz      
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=5):
        answers = []
        ans = { 'fraction': 0 }
        [_mr, _inv_matrix_feedback, _ans] = quiz.data
        if _mr.is_singular:     
            if _mr.rows != _mr.cols:
                ans['feedback'] = r'逆行列が存在するための必要条件の1つが，正方行列であることです。'
                ans['data'] = set(linear_algebra_next_generation.flatten_list_all(_mr.get_inverse_matrix()))
                if ans['data'] != _ans:
                    answers.append(dict(ans))
                ans['data'] = set([-_elem for _elem in ans['data']])
                if ans['data'] != _ans:
                    answers.append(dict(ans))
                ans['data'] = set(linear_algebra_next_generation.flatten_list_all([_row[-_mr.cols:] for _row in _mr.get_matrix_extended_rref()]))
                if ans['data'] != _ans:
                    answers.append(dict(ans))
                ans['data'] = set([-_elem for _elem in ans['data']])
                if ans['data'] != _ans:
                    answers.append(dict(ans))
            else:
                ans['feedback'] = r'正方行列の定義を確認しましょう。この行列は正方行列です。'
                ans['data'] = r'そもそも正方行列ではない。'
                answers.append(dict(ans))
                ans['feedback'] = r'階数の定義を確認してください。階数が行数と一致しなければ逆行列は存在しません。'
                ans['data'] = set(linear_algebra_next_generation.flatten_list_all(_mr.get_inverse_matrix()))
                if ans['data'] != _ans:
                    answers.append(dict(ans))
                ans['data'] = set([-_elem for _elem in ans['data']])
                if ans['data'] != _ans:
                    answers.append(dict(ans))
        else:     
            ans['feedback'] = r'正方行列の定義を確認しましょう。この行列は正方行列です。' + _inv_matrix_feedback
            ans['data'] = r'そもそも正方行列ではない。'
            answers.append(dict(ans)) 
            ans['feedback'] = r'階数の定義を確認してください。階数と行数は一致しています。' + _inv_matrix_feedback
            ans['data'] = r'階数が行数と異なり正則ではない。'
            answers.append(dict(ans)) 
            ans['feedback'] = r'逆行列の計算では，掃き出したままが結果となります。' + _inv_matrix_feedback
            ans['data'] = set(linear_algebra_next_generation.flatten_list_all([[-_elem for _elem in _row] for _row in _mr.get_inverse_matrix()]))
            if ans['data'] != _ans:
                answers.append(dict(ans))
        ans['feedback'] = r'逆行列の定義を確認しましょう。符号反転しただけでは一般に逆行列にはなりません。' + _inv_matrix_feedback       
        ans['data'] = set(linear_algebra_next_generation.flatten_list_all([[-_elem for _elem in _row] for _row in _mr.get_matrix()]))
        if ans['data'] != _ans:
            answers.append(dict(ans))                     
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_mr, _inv_matrix_feedback, _ans] = quiz.data
        _text = r'次の行列の逆行列を求め，最も適切な説明を選択してください。<br />'
        _text += _mr.str_matrix(is_latex_closure=True)
        return _text
    def answer_text(self, ans):
        if isinstance(ans['data'], set):
            _text = r'逆行列は存在し，その各成分を集めた集合は，\( \{'
            for _elem in ans['data']:
                _text += sympy.latex(_elem) + r',\;'
            _text = _text[:-3]
            _text += r'\} \)に等しい。'
            return _text
        return ans['data']


# In[13]:


if __name__ == "__main__":
    q = inverse_matrix()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=25)


# In[14]:


if __name__ == "__main__":
    pass
    #qz.save('inverse_matrix.xml')


# ## composition of permutations

# In[65]:


class composition_permutations(core.Question):
    name = '置換の合成（置換表現の理解の確認）'
    def __init__(self, dmin=2, dmax=4, nmin=2, nmax=3):
        # 集合のサイズの範囲
        self.dim_min = dmin
        self.dim_max = dmax
        # 置換の個数の範囲
        self.num_min = nmin
        self.num_max = nmax        
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='置換の合成', quiz_number=_quiz_number)
        _dim = random.randint(self.dim_min, self.dim_max)
        _num = random.randint(self.num_min, self.num_max)
        _perms = [sympy.combinatorics.Permutation(random.sample(range(_dim),_dim)) for i in range(_num)]
        quiz.quiz_identifier = hash(str(_dim) + str(_num) + str(_perms))
        # 正答の選択肢の生成
        _comp = functools.reduce(lambda x, y: y*x, _perms)
        ans = { 'fraction': 100, 'data': _comp }
        quiz.answers.append(ans)
        quiz.data = [_dim, _num, _perms, _comp]
        return quiz      
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=5):
        answers = []
        ans = { 'fraction': 0 }
        [_dim, _num, _perms, _comp] = quiz.data
        ans['feedback'] = r'合成写像においては，右側の写像が先に適用されます。逆にしないでください。'
        ans['data'] = functools.reduce(lambda x, y: x*y, _perms)
        if ans['data'] != _comp:
            answers.append(dict(ans))
        ans['feedback'] = r'置換であって，行列ではありません。和ではなく合成写像を求めてください。'
        _matrix = sympy.zeros(rows=2, cols=_dim)
        for _perm in _perms:
            _matrix += self._matrix_form(_perm)
        if self._matrix_form(_comp).tolist()[1] != _matrix.tolist()[1]:
            ans['data'] = _matrix
            answers.append(dict(ans))
        ans['feedback'] = r'置換であって，行列ではありません。積ではなく合成写像を求めてください。'
        _matrix = sympy.ones(rows=2, cols=_dim)
        for _perm in _perms:
            _each_perm = self._matrix_form(_perm)
            for _i in range(2):
                for _j in range(_dim):
                    _matrix[_i,_j] *= _each_perm[_i,_j]
        if self._matrix_form(_comp).tolist()[1] != _matrix.tolist()[1]:
            ans['data'] = _matrix
            answers.append(dict(ans))
        ans['feedback'] = r'合成してください。登場するどれかの写像になるわけではありません。'
        for _perm in _perms:
            ans['data'] = _perm
            if ans['data'] != _comp:
                answers.append(dict(ans))
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def _matrix_form(self, perm):
        if isinstance(perm, sympy.combinatorics.permutations.Permutation):
            _before = [_e + 1 for _e in range(perm.size)]
            _after = [_e + 1 for _e in perm.array_form]
            return sympy.Matrix([_before,_after])
        elif isinstance(perm, sympy.Matrix):
            return perm
        else:
            return sympy.Matrix([sorted(perm),perm])
    def question_text(self, quiz):
        [_dim, _num, _perms, _comp] = quiz.data
        _text = r'次の置換の合成（\( ' 
        for i in range(_num):
            _text += r'\sigma_{' + str(i+1) + r'}\circ'
        _text = _text[:-5]
        _text += r' \)）を求めてください。<br />\( '
        for i in range(_num):
            _text += r'\sigma_{' + str(i+1) + r'}='
            _text += sympy.latex(self._matrix_form(_perms[i]), mat_delim='', mat_str='pmatrix') 
            _text += r',\;'        
        _text = _text[:-3]        
        return _text + r' \)'
    def answer_text(self, ans):
        _text = r'\( '
        _text += sympy.latex(self._matrix_form(ans['data']), mat_delim='', mat_str='pmatrix')        
        return _text + r' \)'


# In[66]:


if __name__ == "__main__":
    q = composition_permutations()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=25)


# In[67]:


if __name__ == "__main__":
    pass
    #qz.save('composition_permutations.xml')


# ## inverse of permutations

# In[96]:


class inverse_permutations(core.Question):
    name = '逆置換（置換表現の理解の確認）'
    def __init__(self, dmin=3, dmax=6):
        # 集合のサイズの範囲
        self.dim_min = dmin
        self.dim_max = dmax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='逆置換', quiz_number=_quiz_number)
        _dim = random.randint(self.dim_min, self.dim_max)
        _perm = sympy.combinatorics.Permutation(random.sample(range(_dim),_dim))
        quiz.quiz_identifier = hash(str(_dim) + str(_perm))
        # 正答の選択肢の生成
        _inv = _perm**(-1)
        ans = { 'fraction': 100, 'data': _inv }
        quiz.answers.append(ans)
        quiz.data = [_dim, _perm, _inv]
        return quiz      
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=5):
        answers = []
        ans = { 'fraction': 0 }
        [_dim, _perm, _inv] = quiz.data
        ans['feedback'] = r'合成写像が恒等置換になるのが，逆置換です。それ自身とは限りません。'
        ans['data'] = _perm
        if ans['data'] != _inv:
            answers.append(dict(ans))
        ans['feedback'] = r'合成写像が恒等置換になるのが，逆置換です。単純に順序を逆にしたものではありません。'
        ans['data'] = sympy.combinatorics.Permutation(list(reversed(_perm.array_form)))
        if ans['data'] != _inv:
            answers.append(dict(ans))
        nm_loop = 0
        while len(answers) < size:
            nm_loop += 1
            ans['feedback'] = r'合成写像が恒等置換になるのが，逆置換です。定義に基づけばすぐにわかります。'
            ans['data'] = sympy.combinatorics.Permutation(random.sample(range(_dim),_dim))
            if ans['data'] != _inv:
                answers.append(dict(ans))
            answers = common.answer_union(answers)
            if nm_loop >= sympy.factorial(_dim):
                break
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def _matrix_form(self, perm):
        if isinstance(perm, sympy.combinatorics.permutations.Permutation):
            _before = [_e + 1 for _e in range(perm.size)]
            _after = [_e + 1 for _e in perm.array_form]
            return sympy.Matrix([_before,_after])
        elif isinstance(perm, sympy.Matrix):
            return perm
        else:
            return sympy.Matrix([sorted(perm),perm])
    def question_text(self, quiz):
        [_dim, _perm, _inv] = quiz.data
        _text = r'次の置換の逆置換（\( \sigma^{-1} \)）を求めてください。<br />\( \sigma='
        _text += sympy.latex(self._matrix_form(_perm), mat_delim='', mat_str='pmatrix') 
        return _text + r' \)'
    def answer_text(self, ans):
        _text = r'\( '
        _text += sympy.latex(self._matrix_form(ans['data']), mat_delim='', mat_str='pmatrix')        
        return _text + r' \)'


# In[97]:


if __name__ == "__main__":
    q = inverse_permutations()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=25)


# In[98]:


if __name__ == "__main__":
    pass
    #qz.save('inverse_permutations.xml')


# ## cycle form of permutations

# In[132]:


class cycle_form_permutations(core.Question):
    name = '巡回置換への変換（置換表現の理解の確認）'
    def __init__(self, dmin=3, dmax=6):
        # 集合のサイズの範囲
        self.dim_min = dmin
        self.dim_max = dmax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='巡回置換への変換', quiz_number=_quiz_number)
        _dim = random.randint(self.dim_min, self.dim_max)
        _perm = sympy.combinatorics.Permutation(random.sample(range(_dim),_dim))
        quiz.quiz_identifier = hash(str(_dim) + str(_perm))
        # 正答の選択肢の生成
        _cf = _perm.cyclic_form
        ans = { 'fraction': 100, 'data': _cf }
        quiz.answers.append(ans)
        quiz.data = [_dim, _perm, _cf]
        return quiz      
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=5):
        answers = []
        ans = { 'fraction': 0 }
        [_dim, _perm, _cf] = quiz.data
        ans['feedback'] = r'順番に要素を辿り，巡回置換に直してください。そのまま取り出したものではありません。'
        ans['data'] = [_perm.array_form]
        if sympy.combinatorics.Permutation(ans['data'],size=_dim) != _perm:
            answers.append(dict(ans))
        nm_loop = 0
        while len(answers) < size:
            nm_loop += 1
            ans['feedback'] = r'順番に要素を辿り，巡回置換に直してください。定義に基づけばすぐにわかります。'
            _incorrect_perm = sympy.combinatorics.Permutation(random.sample(range(_dim),_dim))
            ans['data'] = _incorrect_perm.cyclic_form
            if _incorrect_perm != _perm:
                answers.append(dict(ans))
            answers = common.answer_union(answers)
            if nm_loop >= sympy.factorial(_dim):
                break
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def _matrix_form(self, perm):
        if isinstance(perm, sympy.combinatorics.permutations.Permutation):
            _before = [_e + 1 for _e in range(perm.size)]
            _after = [_e + 1 for _e in perm.array_form]
            return sympy.Matrix([_before,_after])
        elif isinstance(perm, sympy.Matrix):
            return perm
        else:
            return sympy.Matrix([sorted(perm),perm])
    def _cycle_form_in_latex(self, perm, mat_delim, mat_str):
        _text = r''
        if len(perm) == 0:
            return r'\varepsilon'
        for _p in perm:
            _text += sympy.latex(sympy.Matrix([[_e + 1 for _e in _p]]), mat_delim=mat_delim, mat_str=mat_str) 
        return _text
    def question_text(self, quiz):
        [_dim, _perm, _cf] = quiz.data
        _text = r'次の置換と同じ置換を選択してください。<br />\( \sigma='
        _text += sympy.latex(self._matrix_form(_perm), mat_delim='', mat_str='pmatrix') 
        return _text + r' \)'
    def answer_text(self, ans):
        _text = r'\( '
        _text += self._cycle_form_in_latex(ans['data'], mat_delim='', mat_str='pmatrix')        
        return _text + r' \)'


# In[133]:


if __name__ == "__main__":
    q = cycle_form_permutations()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=25)


# In[134]:


if __name__ == "__main__":
    pass
    #qz.save('cycle_form_permutations.xml')


# ## transpositions and sign of permutations

# In[145]:


class transpositions_sign_permutations(core.Question):
    name = '互換の積と符号（置換表現の理解の確認）'
    def __init__(self, dmin=3, dmax=6):
        # 集合のサイズの範囲
        self.dim_min = dmin
        self.dim_max = dmax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='互換の積と符号', quiz_number=_quiz_number)
        _dim = random.randint(self.dim_min, self.dim_max)
        _perm = sympy.combinatorics.Permutation(random.sample(range(_dim),_dim))
        quiz.quiz_identifier = hash(str(_dim) + str(_perm))
        # 正答の選択肢の生成
        _cf = _perm.transpositions()
        _sign = sympy.Integer(-1)**len(_cf)
        ans = { 'fraction': 100, 'data': [_cf, _sign] }
        quiz.answers.append(ans)
        quiz.data = [_dim, _perm, _cf, _sign]
        return quiz      
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=5):
        answers = []
        ans = { 'fraction': 0 }
        [_dim, _perm, _cf, _sign] = quiz.data
        ans['feedback'] = r'互換の積で表した際に，互換が偶数個ならば正で，奇数個ならば負です。'
        ans['data'] = [_cf, _sign*sympy.Integer(-1)]
        answers.append(dict(ans))
        if len(_cf) > 1:
            ans['feedback'] = r'互換の積で表してください。そのまま取り出したものではありません。'
            ans['data'] = [[_perm.array_form], -1]
            answers.append(dict(ans))
            ans['data'] = [[_perm.array_form], +1]
            answers.append(dict(ans))
        _incorrect_perm = _perm.cyclic_form
        if len(_cf) != len(_incorrect_perm):
            ans['feedback'] = r'互換の積で表してください。互換でない巡回置換のままではいけません。'
            ans['data'] = [_incorrect_perm, -1]
            answers.append(dict(ans))
            ans['data'] = [_incorrect_perm, +1]
            answers.append(dict(ans))
        ans['feedback'] = r'順番に要素を辿り，巡回置換に直し，更に互換に直してください。定義に基づけばすぐにわかります。'
        _incorrect_perm = sympy.combinatorics.Permutation(random.sample(range(_dim),_dim))
        if _perm != _incorrect_perm:
            ans['data'] = [_incorrect_perm.transpositions(), -1]
            answers.append(dict(ans))
            ans['data'] = [_incorrect_perm.transpositions(), +1]
            answers.append(dict(ans))        
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def _matrix_form(self, perm):
        if isinstance(perm, sympy.combinatorics.permutations.Permutation):
            _before = [_e + 1 for _e in range(perm.size)]
            _after = [_e + 1 for _e in perm.array_form]
            return sympy.Matrix([_before,_after])
        elif isinstance(perm, sympy.Matrix):
            return perm
        else:
            return sympy.Matrix([sorted(perm),perm])
    def _cycle_form_in_latex(self, perm, mat_delim, mat_str):
        _text = r''
        if len(perm) == 0:
            return r'\varepsilon'
        for _p in perm:
            _text += sympy.latex(sympy.Matrix([[_e + 1 for _e in _p]]), mat_delim=mat_delim, mat_str=mat_str) 
        return _text
    def question_text(self, quiz):
        [_dim, _perm, _cf, _sign] = quiz.data
        _text = r'次の置換を互換の積で表し，また符号も正しく選択してください。<br />\( \sigma='
        _text += sympy.latex(self._matrix_form(_perm), mat_delim='', mat_str='pmatrix') 
        return _text + r' \)'
    def answer_text(self, ans):
        _text = r'\( ' + self._cycle_form_in_latex(ans['data'][0], mat_delim='', mat_str='pmatrix')
        _text += r',\;\textrm{sign}(\sigma)='
        if ans['data'][1] > 0:
            _text +=  r'+1 \)'
        else:
            _text +=  r'-1 \)'
        return _text


# In[146]:


if __name__ == "__main__":
    q = transpositions_sign_permutations()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=25)


# In[147]:


if __name__ == "__main__":
    pass
    #qz.save('transpositions_sign_permutations.xml')


# ## determinant by sarrus rule

# In[162]:


class determinant_by_sarrus(core.Question):
    name = '行列式の計算（サラスの方法での計算）'
    def __init__(self, dmin=2, dmax=3, srate=0.75, emin=-3, emax=3):
        # 行列のサイズの範囲
        self.dim_min = dmin
        self.dim_max = dmax
        # 正方行列となる確率
        self.square_rate = srate
        # 各要素の範囲
        self.elem_min = emin
        self.elem_max = emax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='行列式の計算（サラス）', quiz_number=_quiz_number)
        _is_square = True if random.random() <= self.square_rate else False
        _rows = random.randint(self.dim_min, self.dim_max)
        _cols = _rows
        while _cols == _rows and not _is_square:
            _cols = random.randint(self.dim_min, self.dim_max)
        _matrix = [[random.randint(self.elem_min, self.elem_max) for _j in range(_cols)] for _i in range(_rows)]
        quiz.quiz_identifier = hash(str(_is_square) + str(_rows) + str(_cols) + str(_matrix))
        # 正答の選択肢の生成
        _det = r'行列式は定義されない。'
        if _is_square:
            _det = sympy.Matrix(_matrix).det()
        ans = { 'fraction': 100, 'data': _det }
        quiz.answers.append(ans)
        quiz.data = [_is_square, _rows, _cols, _matrix, _det]
        return quiz      
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=5):
        answers = []
        ans = { 'fraction': 0 }
        [_is_square, _rows, _cols, _matrix, _det] = quiz.data
        if _is_square:
            ans['feedback'] = r'与えられた行列は正方行列です。行列式は定義されます。'
            ans['data'] = r'行列式は定義されない。'
            answers.append(dict(ans))
        _matrix = [[_matrix[_i][_j] for _j in range(min(_rows,_cols))] for _i in range(min(_rows,_cols))]
        if isinstance(_det, str):
            _det = sympy.sqrt(821641) # any number that may be different from the determinant
        ans['feedback'] = r'行列式は正方行列に対してのみ定義されます。よく確認してください。'
        _incorrect_det = -sympy.Matrix(_matrix).det()
        if _is_square:
            ans['feedback'] = r'サラスの方法では，左上から右下へが正で，右上から左下が負です。'
        if _incorrect_det != _det:
            ans['data'] = _incorrect_det
            answers.append(dict(ans))
        _incorrect_det = 0
        if _is_square:
            ans['feedback'] = r'サラスの方法を確認し，丁寧に計算してください。'
        if _incorrect_det != _det:
            ans['data'] = _incorrect_det
            answers.append(dict(ans))
        if min(_rows,_cols) == 3:
            if _is_square:
                ans['feedback'] = r'サラスの方法では，2次正方行列と異なり，クロスの積だけではありません。'
            _incorrect_det = _matrix[0][0]*_matrix[1][1]*_matrix[2][2] - _matrix[0][2]*_matrix[1][1]*_matrix[2][0]
            if _incorrect_det != _det:
                ans['data'] = _incorrect_det
                answers.append(dict(ans))
            _incorrect_det *= -1
            if _incorrect_det != _det:
                ans['data'] = _incorrect_det
                answers.append(dict(ans))
        while len(answers) < size:
            _abs_det_max = max(abs(self.elem_min),abs(self.elem_max))**min(_rows,_cols)
            _incorrect_det = random.randint(-_abs_det_max, _abs_det_max)
            if _is_square:
                ans['feedback'] = r'サラスの方法を確認し，丁寧に計算してください。'
            if _incorrect_det != _det:
                ans['data'] = _incorrect_det
                answers.append(dict(ans))
            if -_incorrect_det != _det:
                ans['data'] = -_incorrect_det
                answers.append(dict(ans))
            answers = common.answer_union(answers)
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_is_square, _rows, _cols, _matrix, _det] = quiz.data
        _text = r'次の行列の行列式が定義されるならば，サラスの方法で求めてください。<br />\( '
        _text += sympy.latex(sympy.Matrix(_matrix), mat_delim='', mat_str='pmatrix') 
        return _text + r' \)'
    def answer_text(self, ans):
        if isinstance(ans['data'], str):
            return ans['data']
        else:
            return r'\( ' + sympy.latex(ans['data'], mat_delim='', mat_str='pmatrix') + r' \)'


# In[163]:


if __name__ == "__main__":
    q = determinant_by_sarrus()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=25)


# In[164]:


if __name__ == "__main__":
    pass
    #qz.save('determinant_by_sarrus.xml')


# ## determinant by sarrus rule + alpha

# In[15]:


class determinant_by_sarrus_plus_alpha(core.Question):
    """
    non-zero leading column and scalar multiplied row, and sarrus.
    """
    name = '行列式の計算（スカラー倍とサイズダウン後にサラスの方法での計算）'
    def __init__(self, dmin=4, dmax=4, srate=0.9, emin=-3, emax=3):
        # 行列のサイズの範囲
        self.dim_min = dmin
        self.dim_max = dmax
        # 正方行列となる確率
        self.square_rate = srate
        # 各要素の範囲
        self.elem_min = emin
        self.elem_max = emax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='行列式の計算（少し変形＋サラス）', quiz_number=_quiz_number)
        _is_square = True if random.random() <= self.square_rate or self.dim_min == self.dim_max else False
        _rows = random.randint(self.dim_min, self.dim_max)
        _cols = _rows
        while _cols == _rows and not _is_square:
            _cols = random.randint(self.dim_min, self.dim_max)
        _matrix = [[random.randint(self.elem_min, self.elem_max) for _j in range(_cols)] for _i in range(_rows)]
        _tr = random.choice(range(_rows))
        _ts = random.choice([-100,-10,10,100])
        _matrix =[[_matrix[_i][_j] if _i != _tr else _ts*_matrix[_i][_j] for _j in range(_cols)] for _i in range(_rows)]
        _le = linear_algebra_next_generation.nonzero_randint(self.elem_min, self.elem_max)
        _matrix[0][0] = _le
        _matrix =[[_matrix[_i][_j] if _j != 0 or _i == 0 else 0 for _j in range(_cols)] for _i in range(_rows)]        
        quiz.quiz_identifier = hash(str(_is_square) + str(_rows) + str(_cols) + str(_matrix))
        # 正答の選択肢の生成
        _det = r'行列式は定義されない。'
        if _is_square:
            _det = sympy.Matrix(_matrix).det()
        ans = { 'fraction': 100, 'data': _det }
        quiz.answers.append(ans)
        quiz.data = [_is_square, _rows, _cols, _matrix, _det, _le]
        return quiz      
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=5):
        answers = []
        ans = { 'fraction': 0 }
        [_is_square, _rows, _cols, _matrix, _det, _le] = quiz.data
        _sub_matrix = [[_matrix[_i][_j] for _j in range(1,_cols)] for _i in range(1,_rows)]
        if _is_square:
            ans['feedback'] = r'与えられた行列は正方行列です。行列式は定義されます。'
            ans['data'] = r'行列式は定義されない。'
            answers.append(dict(ans))
        _sub_matrix = [[_sub_matrix[_i][_j] for _j in range(min(_rows,_cols)-1)] for _i in range(min(_rows,_cols)-1)]
        if isinstance(_det, str):
            _det = sympy.sqrt(821641) # any number that may be different from the determinant
        ans['feedback'] = r'行列式は正方行列に対してのみ定義されます。よく確認してください。'
        _incorrect_det = -_le*sympy.Matrix(_sub_matrix).det()
        if _is_square:
            ans['feedback'] = r'サラスの方法では，左上から右下へが正で，右上から左下が負です。'
        if _incorrect_det != _det:
            ans['data'] = _incorrect_det
            answers.append(dict(ans))
        _incorrect_det = sympy.Matrix(_sub_matrix).det()
        if _is_square:
            ans['feedback'] = r'サイズダウンしたときの左上の成分を忘れていませんか。'
        if _incorrect_det != _det:
            ans['data'] = _incorrect_det
            answers.append(dict(ans))
        if -_incorrect_det != _det:
            ans['data'] = -_incorrect_det
            answers.append(dict(ans))
        _incorrect_det = 0
        if _is_square:
            ans['feedback'] = r'サラスの方法を確認し，丁寧に計算してください。'
        if _incorrect_det != _det:
            ans['data'] = _incorrect_det
            answers.append(dict(ans))
        if min(_rows,_cols)-1 == 3:
            if _is_square:
                ans['feedback'] = r'サラスの方法では，2次正方行列と異なり，クロスの積だけではありません。'
            _incorrect_det = _sub_matrix[0][0]*_sub_matrix[1][1]*_sub_matrix[2][2] - _sub_matrix[0][2]*_sub_matrix[1][1]*_sub_matrix[2][0]
            if _incorrect_det != _det:
                ans['data'] = _incorrect_det
                answers.append(dict(ans))
            _incorrect_det *= -1
            if _incorrect_det != _det:
                ans['data'] = _incorrect_det
                answers.append(dict(ans))
        while len(answers) < size:
            _abs_det_max = max(abs(self.elem_min),abs(self.elem_max))**min(_rows,_cols)
            _incorrect_det = random.randint(-_abs_det_max, _abs_det_max)*random.choice([1,10])
            if _is_square:
                ans['feedback'] = r'サラスの方法を確認し，丁寧に計算してください。'
            if _incorrect_det != _det:
                ans['data'] = _incorrect_det
                answers.append(dict(ans))
            if -_incorrect_det != _det:
                ans['data'] = -_incorrect_det
                answers.append(dict(ans))
            answers = common.answer_union(answers)
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_is_square, _rows, _cols, _matrix, _det, _le] = quiz.data
        _text = r'次の行列の行列式が定義されるならば，サラスの方法で求めてください。<br />\( '
        _text += sympy.latex(sympy.Matrix(_matrix), mat_delim='', mat_str='pmatrix') 
        return _text + r' \)'
    def answer_text(self, ans):
        if isinstance(ans['data'], str):
            return ans['data']
        else:
            return r'\( ' + sympy.latex(ans['data'], mat_delim='', mat_str='pmatrix') + r' \)'


# In[16]:


if __name__ == "__main__":
    q = determinant_by_sarrus_plus_alpha()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=25)


# In[17]:


if __name__ == "__main__":
    pass
    #qz.save('determinant_by_sarrus_plus_alpha.xml')


# ## same determinant by row operations

# In[28]:


class same_determinant_by_row_ops(core.Question):
    name = '同じ行列式を持つ行列（行の基本変形と行列式）'
    def __init__(self, dmin=3, dmax=4, emin=-3, emax=3):
        # 行列のサイズの範囲
        self.dim_min = dmin
        self.dim_max = dmax
        # 各要素の範囲
        self.elem_min = emin
        self.elem_max = emax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='同じ行列式を持つ行列', quiz_number=_quiz_number)
        _dim = random.randint(self.dim_min, self.dim_max)
        _matrix = [[random.randint(self.elem_min, self.elem_max) for _j in range(_dim)] for _i in range(_dim)]
        while sympy.Matrix(_matrix).det() == 0:
            _matrix = [[random.randint(self.elem_min, self.elem_max) for _j in range(_dim)] for _i in range(_dim)]
        # 正答の選択肢の生成
        _sympy_matrix = sympy.Matrix(_matrix)
        _op_types = ['n->kn', 'n<->m', 'n->n+km']
        _op = random.choice(_op_types)
        if _op == 'n->n+km':
            _k = 1 if random.random() < 0.5 else -1
        else:
            _k = linear_algebra_next_generation.nonzeroone_randint(self.elem_min, self.elem_max)
        _row = random.choice(range(_dim))
        _row2 = _row
        while _row == _row2:
            _row2 = random.choice(range(_dim))
        _ans_matrix = _sympy_matrix.elementary_row_op(op=_op, row=_row, k=_k, row2=_row2).tolist()
        if _op == 'n<->m':
            _sc = -1
        elif _op == 'n->kn':
            _sc = 1/sympy.Integer(_k)
        else:
            _sc = 1
        quiz.quiz_identifier = hash(str(_dim) + str(_matrix) + str(_ans_matrix))
        ans = { 'fraction': 100, 'data': [_sc, _ans_matrix] }
        quiz.answers.append(ans)
        quiz.data = [_dim, _matrix, _ans_matrix, _sc, _op]
        return quiz      
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=5):
        answers = []
        ans = { 'fraction': 0 }
        [_dim, _matrix, _ans_matrix, _sc, _op] = quiz.data
        _det = sympy.Matrix(_matrix).det()
        _ans_det = sympy.Matrix(_ans_matrix).det()
        if _op == 'n<->m':
            if _det != _ans_det:
                ans['feedback'] = r'行の交換を行うと符号は反転します。'
                ans['data'] = [1, _ans_matrix]
                answers.append(dict(ans))
        elif _det != -_ans_det:
            ans['feedback'] = r'符号が反転するのは，行の交換を行ったときだけです。'
            ans['data'] = [-1, _ans_matrix]
            answers.append(dict(ans))
        if _op == 'n->kn':
            if _det != _ans_det:
                ans['feedback'] = r'行のスカラー倍を行うと行列式はその分だけ変化します。'
                ans['data'] = [1, _ans_matrix]
                answers.append(dict(ans))
            if _det != -_ans_det:
                ans['feedback'] = r'行のスカラー倍を行うと行列式はその分だけ変化します。'
                ans['data'] = [-1, _ans_matrix]
                answers.append(dict(ans))
        while len(answers) < size:
            _sympy_matrix = sympy.Matrix(_matrix)
            _inc_op_types = ['n->kn', 'n<->m', 'n->n+km']
            _inc_op = random.choice(_inc_op_types)
            if _inc_op == 'n->n+km':
                _k = 1 if random.random() < 0.5 else -1
            else:
                _k = linear_algebra_next_generation.nonzeroone_randint(self.elem_min, self.elem_max)
            _row = random.choice(range(_dim))
            _row2 = _row
            while _row == _row2:
                _row2 = random.choice(range(_dim))
            _inc_ans_matrix = _sympy_matrix.elementary_row_op(op=_inc_op, row=_row, k=_k, row2=_row2).tolist()
            _inc_ans_det = sympy.Matrix(_inc_ans_matrix).det()
            if _inc_op == 'n<->m':
                if _det != _inc_ans_det:
                    ans['feedback'] = r'行の交換を行うと符号は反転します。'
                    ans['data'] = [1, _inc_ans_matrix]
                    answers.append(dict(ans))
                if _det != _inc_ans_det/sympy.Integer(_k):
                    ans['feedback'] = r'行の交換を行うと符号は反転します。'
                    ans['data'] = [1/sympy.Integer(_k), _inc_ans_matrix]
                    answers.append(dict(ans))
            elif _inc_op == 'n->kn':
                if _det != _inc_ans_det:
                    ans['feedback'] = r'行のスカラー倍を行うと行列式はその分だけ変化します。'
                    ans['data'] = [1, _inc_ans_matrix]
                    answers.append(dict(ans))
                if _det != -_inc_ans_det:
                    ans['feedback'] = r'行のスカラー倍を行うと行列式はその分だけ変化します。'
                    ans['data'] = [-1, _inc_ans_matrix]
                    answers.append(dict(ans))
            else:
                if _det != _inc_ans_det/sympy.Integer(_k):
                    ans['feedback'] = r'ある行の何倍かを他の行に加えても行列式は変化しません。'
                    ans['data'] = [1/sympy.Integer(_k), _inc_ans_matrix]
                    answers.append(dict(ans))
                if _det != -_inc_ans_det:
                    ans['feedback'] = r'ある行の何倍かを他の行に加えても行列式は変化しません。'
                    ans['data'] = [-1, _inc_ans_matrix]
                    answers.append(dict(ans))
            answers = common.answer_union(answers)
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_dim, _matrix, _ans_matrix, _sc, _op] = quiz.data
        _text = r'次の行列と同じ行列式を持つ行列を選んでください（行の基本変形を活用してください）。<br />\( '
        _text += sympy.latex(sympy.Matrix(_matrix), mat_delim='', mat_str='pmatrix') 
        return _text + r' \)'
    def answer_text(self, ans):
        if ans['data'][0] == 1:
            return r'\( ' + sympy.latex(sympy.Matrix(ans['data'][1]), mat_delim='', mat_str='pmatrix') + r' \)'
        elif ans['data'][0] == -1:
            return r'\( -' + sympy.latex(sympy.Matrix(ans['data'][1]), mat_delim='', mat_str='pmatrix') + r' \)'
        else:
            return r'\( ' + sympy.latex(ans['data'][0]) + sympy.latex(sympy.Matrix(ans['data'][1]), mat_delim='', mat_str='pmatrix') + r' \)'


# In[29]:


if __name__ == "__main__":
    q = same_determinant_by_row_ops()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=25)


# In[30]:


if __name__ == "__main__":
    pass
    #qz.save('same_determinant_by_row_ops.xml')


# ## determinant of nearly row echelon form

# In[42]:


class determinant_of_nearly_ref(core.Question):
    name = '行列式の計算（ほぼ階段行列の場合）'
    def __init__(self, dmin=4, dmax=4, emin=-3, emax=3):
        # 行列のサイズの範囲
        self.dim_min = dmin
        self.dim_max = dmax
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='行列式の計算（ほぼ階段行列）', quiz_number=_quiz_number)
        _dim = random.randint(self.dim_min,self.dim_max)
        _mr = linear_algebra_next_generation.MatrixInRowEchelonForm()
        _mr.set_dimension_range(_dim, _dim)
        _mr.set_element_range(self.elem_min,self.elem_max)
        _mr.set_zero_vector_ratio(0.0)
        _mr.generate(is_swap_only=True)
        quiz.quiz_identifier = hash(_mr)
        # 正答の選択肢の生成
        _det = sympy.Matrix(_mr.get_matrix()).det()
        quiz.data = [_mr, _det]
        ans = { 'fraction': 100, 'data': _det }
        quiz.answers.append(ans)
        return quiz      
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=5):
        answers = []
        ans = { 'fraction': 0 }
        [_mr, _det] = quiz.data
        ans['feedback'] = r'行の基本変形で三角行列に変形しましょう。三角行列の行列式は対角成分の積となります。'
        if _det != 0:
            ans['data'] = 0
            answers.append(dict(ans)) 
            ans['data'] = -_det
            answers.append(dict(ans))
        _factors = [[_p for _i in range(_e)] for _p,_e in abs(_det).factors().items()]
        _factors = linear_algebra_next_generation.flatten_list_all(_factors)
        while len(_factors) < 2:
            _factors.append(abs(linear_algebra_next_generation.nonzeroone_randint(self.elem_min,self.elem_max)))
        for _i in range(len(_factors)):
            _incorrect_det = functools.reduce(lambda x, y: x*y, _factors[:_i] + _factors[_i+1:])
            if _incorrect_det != _det:
                ans['data'] = _incorrect_det
                answers.append(dict(ans)) 
            if -_incorrect_det != _det:
                ans['data'] = -_incorrect_det
                answers.append(dict(ans))
        answers = common.answer_union(answers)
        while len(answers) < size:
            _inc_factors = [_e if random.random() < 0.3 else abs(linear_algebra_next_generation.nonzeroone_randint(self.elem_min,self.elem_max)) for _e in _factors]
            _incorrect_det = functools.reduce(lambda x, y: x*y, _inc_factors)
            if _incorrect_det != _det:
                ans['data'] = _incorrect_det
                answers.append(dict(ans)) 
            if -_incorrect_det != _det:
                ans['data'] = -_incorrect_det
                answers.append(dict(ans))
            answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_mr, _det] = quiz.data
        _text = r'次の行列の行列式を求めてください。<br />'
        _text += r'\( ' + sympy.latex(sympy.Matrix(_mr.get_matrix()), mat_delim='', mat_str='pmatrix') + r' \)'
        return _text
    def answer_text(self, ans):
        return r'\( ' + sympy.latex(ans['data']) + r' \)'


# In[43]:


if __name__ == "__main__":
    q = determinant_of_nearly_ref()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=25)


# In[44]:


if __name__ == "__main__":
    pass
    #qz.save('determinant_of_nearly_ref.xml')


# ## minor expansion

# In[83]:


class minor_expansion(core.Question):
    name = '正しい余因子展開（余因子展開の公式の確認）'
    def __init__(self, dmin=3, dmax=4, emin=-3, emax=3):
        # 行列のサイズの範囲
        self.dim_min = dmin
        self.dim_max = dmax
        # 各要素の範囲
        self.elem_min = emin
        self.elem_max = emax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='正しい余因子展開', quiz_number=_quiz_number)
        _dim = random.randint(self.dim_min, self.dim_max)
        _matrix = [[random.randint(self.elem_min, self.elem_max) for _j in range(_dim)] for _i in range(_dim)]
        _is_row = True if random.random() < 0.5 else False
        _rowcol = random.choice(range(_dim))
        quiz.quiz_identifier = hash(str(_dim) + str(_matrix) + str(_is_row) + str(_rowcol))
        # 正答の選択肢の生成
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
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=5):
        answers = []
        ans = { 'fraction': 0 }
        [_dim, _matrix, _is_row, _rowcol, _ans] = quiz.data
        _incorrect_rowcol = random.choice(list(set(range(_dim))-set([_rowcol])))
        _incorrect_ans = self._make_expansion(_is_row, _rowcol, _matrix, _rowcolM=_incorrect_rowcol)
        if _incorrect_ans != _ans:
            ans['feedback'] = r'展開する行や列に注意してください。'
            ans['data'] = _incorrect_ans
            answers.append(dict(ans))
        _incorrect_ans = self._make_expansion(not _is_row, _rowcol, _matrix, _rowcolM=_incorrect_rowcol)
        if _incorrect_ans != _ans:
            ans['feedback'] = r'展開する行や列に注意してください。また，行と列で余因子展開の公式は異なります。注意してください。'
            ans['data'] = _incorrect_ans
            answers.append(dict(ans))
        if _dim > 2:
            _incorrect_rowcol = random.choice(list(set(range(_dim))-set([_rowcol,_incorrect_rowcol])))
            _incorrect_ans = self._make_expansion(_is_row, _rowcol, _matrix, _rowcolM=_incorrect_rowcol)
            if _incorrect_ans != _ans:
                ans['feedback'] = r'展開する行や列に注意してください。'
                ans['data'] = _incorrect_ans
                answers.append(dict(ans))
            _incorrect_ans = self._make_expansion(not _is_row, _rowcol, _matrix, _rowcolM=_incorrect_rowcol)
            if _incorrect_ans != _ans:
                ans['feedback'] = r'展開する行や列に注意してください。また，行と列で余因子展開の公式は異なります。注意してください。'
                ans['data'] = _incorrect_ans
                answers.append(dict(ans))
        _incorrect_rowcol = random.choice(list(set(range(_dim))-set([_rowcol])))
        _incorrect_ans = self._make_expansion(_is_row, _incorrect_rowcol, _matrix)
        if _incorrect_ans != _ans:
            ans['feedback'] = r'展開する行や列に注意してください。'
            ans['data'] = _incorrect_ans
            answers.append(dict(ans))
        _incorrect_ans = self._make_expansion(not _is_row, _incorrect_rowcol, _matrix)
        if _incorrect_ans != _ans:
            ans['feedback'] = r'展開する行や列に注意してください。また，行と列で余因子展開の公式は異なります。注意してください。'
            ans['data'] = _incorrect_ans
            answers.append(dict(ans))
        _incorrect_ans = self._make_expansion(not _is_row, _rowcol, _matrix)
        if _incorrect_ans != _ans:
            ans['feedback'] = r'行と列で余因子展開の公式は異なります。注意してください。'
            ans['data'] = _incorrect_ans
            answers.append(dict(ans))
        _incorrect_ans = self._make_expansion(_is_row, _rowcol, _matrix, _base=1)
        if _incorrect_ans != _ans:
            ans['feedback'] = r'符号が変化しながら和を取りますので，符号の取り扱いに注意してください。'
            ans['data'] = _incorrect_ans
            answers.append(dict(ans))            
        _incorrect_ans = self._make_expansion(not _is_row, _rowcol, _matrix, _base=1)
        if _incorrect_ans != _ans:
            ans['feedback'] = r'符号が変化しながら和を取りますので，符号の取り扱いに注意してください。また，行と列で余因子展開の公式は異なります。'
            ans['data'] = _incorrect_ans
            answers.append(dict(ans))
        _incorrect_ans = self._make_expansion(_is_row, _rowcol, _matrix, _padding=1)
        if _incorrect_ans != _ans:
            ans['feedback'] = r'符号がおかしいです。符号の取り扱いに注意してください。'
            ans['data'] = _incorrect_ans
            answers.append(dict(ans))            
        _incorrect_ans = self._make_expansion(not _is_row, _rowcol, _matrix, _padding=1)
        if _incorrect_ans != _ans:
            ans['feedback'] = r'符号がおかしいです。符号の取り扱いに注意してください。また，行と列で余因子展開の公式は異なります。'
            ans['data'] = _incorrect_ans
            answers.append(dict(ans))            
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_dim, _matrix, _is_row, _rowcol, _ans] = quiz.data
        _text = r'次の行列式を，第' + str(_rowcol+1)
        if _is_row:
            _text += r'行'
        else:
            _text += r'列'
        _text += r'で余因子展開した式を選んでください。<br />\( '
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


# In[84]:


if __name__ == "__main__":
    q = minor_expansion()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=25)


# In[85]:


if __name__ == "__main__":
    pass
    #qz.save('minor_expansion.xml')


# ## determinant of general matrix

# In[100]:


class determinant_of_general_matrix(core.Question):
    name = '行列式の計算（一般の行列の場合）'
    def __init__(self, dmin=4, dmax=5, emin=-3, emax=3):
        # 行列のサイズの範囲
        self.dim_min = dmin
        self.dim_max = dmax
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='行列式の計算（一般）', quiz_number=_quiz_number)
        _dim = random.randint(self.dim_min,self.dim_max)
        _mr = linear_algebra_next_generation.MatrixInRowEchelonForm()
        _mr.set_dimension_range(_dim, _dim)
        _mr.set_element_range(self.elem_min,self.elem_max)
        _mr.set_zero_vector_ratio(0.0)
        _mr.generate()
        quiz.quiz_identifier = hash(_mr)
        # 正答の選択肢の生成
        _det = sympy.Matrix(_mr.get_matrix()).det()
        quiz.data = [_mr, _det]
        ans = { 'fraction': 100, 'data': _det }
        quiz.answers.append(ans)
        return quiz      
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=5):
        answers = []
        ans = { 'fraction': 0 }
        [_mr, _det] = quiz.data
        ans['feedback'] = r'行や列の基本変形で三角行列に変形するか，0の多い列や行に関して余因子展開をしましょう。余因子展開をすればサラスの方法で計算可能なサイズになりますし，三角行列に変形すれば，その行列式は対角成分の積となります。'
        if _det != 0:
            ans['data'] = 0
            answers.append(dict(ans)) 
            ans['data'] = -_det
            answers.append(dict(ans))
        answers = common.answer_union(answers)
        _factors = [[_p for _i in range(_e)] for _p,_e in abs(_det).factors().items()]
        _factors = linear_algebra_next_generation.flatten_list_all(_factors)
        _max = max(abs(self.elem_min), abs(self.elem_max))
        while len(_factors) < 2:
            _factors.append(random.randint(1, _max))
        while len(answers) < size:
            _inc_factors = [_e if random.random() < 0.3 else random.randint(1, _max) for _e in _factors]
            _incorrect_det = functools.reduce(lambda x, y: x*y, _inc_factors)
            if _incorrect_det != _det:
                ans['data'] = _incorrect_det
                answers.append(dict(ans)) 
            if -_incorrect_det != _det:
                ans['data'] = -_incorrect_det
                answers.append(dict(ans))
            answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_mr, _det] = quiz.data
        _text = r'次の行列の行列式を求めてください。<br />'
        _text += r'\( ' + sympy.latex(sympy.Matrix(_mr.get_matrix()), mat_delim='', mat_str='pmatrix') + r' \)'
        return _text
    def answer_text(self, ans):
        return r'\( ' + sympy.latex(ans['data']) + r' \)'


# In[101]:


if __name__ == "__main__":
    q = determinant_of_general_matrix()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=25)


# In[102]:


if __name__ == "__main__":
    pass
    #qz.save('determinant_of_general_matrix.xml')


# ## adjugate matrix with minors

# In[3]:


class adjugate_matrix_with_minors(core.Question):
    name = '余因子行列（小行列式の計算不要）'
    def __init__(self, dmin=3, dmax=4, emin=-3, emax=3):
        # 行列のサイズの範囲
        self.dim_min = dmin
        self.dim_max = dmax
        # 各要素の範囲
        self.elem_min = emin
        self.elem_max = emax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='余因子行列（計算不要）', quiz_number=_quiz_number)
        _dim = random.randint(self.dim_min, self.dim_max)
        _matrix = [[random.randint(self.elem_min, self.elem_max) for _j in range(_dim)] for _i in range(_dim)]
        _adjugate = self._make_adjugate(_matrix)
        quiz.quiz_identifier = hash(str(_dim) + str(_matrix) + str(_adjugate))
        # 正答の選択肢の生成
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
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=5):
        answers = []
        ans = { 'fraction': 0 }
        [_dim, _matrix, _adjugate] = quiz.data
        ans['feedback'] = r'小行列式から余因子行列を構成する際，符号変化するので注意してください。'
        _incorrect_ans = [[sympy.Matrix(self._make_minor(_matrix,_j,_i)).det()*(1)**(_i+_j) for _j in range(_dim)] for _i in range(_dim)]
        if _incorrect_ans != _adjugate:
            ans['data'] = _incorrect_ans
            answers.append(dict(ans))
        _incorrect_ans = [[sympy.Matrix(self._make_minor(_matrix,_j,_i)).det()*(-1)**(_i+_j+1) for _j in range(_dim)] for _i in range(_dim)]
        if _incorrect_ans != _adjugate:
            ans['data'] = _incorrect_ans
            answers.append(dict(ans))
        ans['feedback'] = r'小行列式から余因子行列を構成する際，符号変化や添字が逆転（転置）するので注意してください。'
        _incorrect_ans = [[sympy.Matrix(self._make_minor(_matrix,_i,_j)).det()*(1)**(_i+_j) for _j in range(_dim)] for _i in range(_dim)]
        if _incorrect_ans != _adjugate:
            ans['data'] = _incorrect_ans
            answers.append(dict(ans))
        _incorrect_ans = [[sympy.Matrix(self._make_minor(_matrix,_i,_j)).det()*(-1)**(_i+_j+1) for _j in range(_dim)] for _i in range(_dim)]
        if _incorrect_ans != _adjugate:
            ans['data'] = _incorrect_ans
            answers.append(dict(ans))
        ans['feedback'] = r'小行列式から余因子行列を構成する際，添字が逆転（転置）するので注意してください。'
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
        _text = r'次の行列の余因子行列を選んでください。<br />\( A='
        _text += sympy.latex(sympy.Matrix(_matrix), mat_delim='', mat_str='pmatrix') + r' \)<br />'
        _text += r'ただし，次の小行列式の情報を活用しても構いません。<br />\( '
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


# In[118]:


if __name__ == "__main__":
    pass
    #qz.save('adjugate_matrix_with_minors.xml')


# ## inverse by adjugate matrix with minors

# In[3]:


class inverse_by_adjugate_matrix_with_minors(core.Question):
    name = '余因子行列による逆行列（小行列式の計算不要）'
    def __init__(self, dmin=3, dmax=4, emin=-3, emax=3):
        # 行列のサイズの範囲
        self.dim_min = dmin
        self.dim_max = dmax
        # 各要素の範囲
        self.elem_min = emin
        self.elem_max = emax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='余因子行列による逆行列', quiz_number=_quiz_number)
        _dim = random.randint(self.dim_min, self.dim_max)        
        _mr = linear_algebra_next_generation.MatrixInRowEchelonForm()
        _mr.set_dimension_range(_dim, _dim)
        _mr.set_element_range(self.elem_min,self.elem_max)
        _mr.set_zero_vector_ratio(0.0)
        _mr.generate(is_swap_only=True)        
        _matrix = _mr.get_matrix()
        _adjugate = self._make_adjugate(_matrix)
        _det = sympy.Matrix(_matrix).det()
        quiz.quiz_identifier = hash(str(_dim) + str(_matrix) + str(_adjugate) + str(_det))
        # 正答の選択肢の生成
        ans = { 'fraction': 100, 'data': [_det, _adjugate] }
        quiz.answers.append(ans)
        quiz.data = [_dim, _matrix, _adjugate, _det]
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
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=5):
        answers = []
        ans = { 'fraction': 0 }
        [_dim, _matrix, _adjugate, _det] = quiz.data
        if _det != 0:
            ans['feedback'] = r'行列式の扱いに注意しましょう。'
            ans['data'] = [-_det, _adjugate]
            answers.append(dict(ans))
        if abs(_det) > 1:
            ans['feedback'] = r'行列式の扱いに注意しましょう。'
            ans['data'] = [1/_det, _adjugate]
            answers.append(dict(ans))
            ans['feedback'] = r'行列式の扱いに注意しましょう。'
            ans['data'] = [-1/_det, _adjugate]
            answers.append(dict(ans))        
        _incorrect_ans = [[sympy.Matrix(self._make_minor(_matrix,_j,_i)).det()*(1)**(_i+_j) for _j in range(_dim)] for _i in range(_dim)]
        if _incorrect_ans != _adjugate:
            ans['feedback'] = r'小行列式から余因子行列を構成する際，符号変化するので注意してください。'
            ans['data'] = [_det, _incorrect_ans]
            answers.append(dict(ans))
            ans['feedback'] = r'小行列式から余因子行列を構成する際，符号変化するので注意してください。また，行列式の扱いにも注意しましょう。'
            ans['data'] = [-_det, _incorrect_ans]
            answers.append(dict(ans))
            if abs(_det) > 1:
                ans['feedback'] = r'小行列式から余因子行列を構成する際，符号変化するので注意してください。また，行列式の扱いにも注意しましょう。'
                ans['data'] = [1/_det, _incorrect_ans]
                answers.append(dict(ans))
                ans['feedback'] = r'小行列式から余因子行列を構成する際，符号変化するので注意してください。また，行列式の扱いにも注意しましょう。'
                ans['data'] = [-1/_det, _incorrect_ans]
                answers.append(dict(ans))                
        _incorrect_ans = [[sympy.Matrix(self._make_minor(_matrix,_i,_j)).det()*(1)**(_i+_j) for _j in range(_dim)] for _i in range(_dim)]
        if _incorrect_ans != _adjugate:
            ans['feedback'] = r'小行列式から余因子行列を構成する際，符号変化や添字が逆転（転置）するので注意してください。'
            ans['data'] = [_det, _incorrect_ans]
            answers.append(dict(ans))
            ans['feedback'] = r'小行列式から余因子行列を構成する際，符号変化や添字が逆転（転置）するので注意してください。また，行列式の扱いにも注意しましょう。'
            ans['data'] = [-_det, _incorrect_ans]
            answers.append(dict(ans))
            if abs(_det) > 1:
                ans['feedback'] = r'小行列式から余因子行列を構成する際，符号変化や添字が逆転（転置）するので注意してください。また，行列式の扱いにも注意しましょう。'
                ans['data'] = [1/_det, _incorrect_ans]
                answers.append(dict(ans))
                ans['feedback'] = r'小行列式から余因子行列を構成する際，符号変化や添字が逆転（転置）するので注意してください。また，行列式の扱いにも注意しましょう。'
                ans['data'] = [-1/_det, _incorrect_ans]
                answers.append(dict(ans))      
        _incorrect_ans = [[sympy.Matrix(self._make_minor(_matrix,_i,_j)).det()*(-1)**(_i+_j) for _j in range(_dim)] for _i in range(_dim)]
        if _incorrect_ans != _adjugate:
            ans['feedback'] = r'小行列式から余因子行列を構成する際，添字が逆転（転置）するので注意してください。'
            ans['data'] = [_det, _incorrect_ans]
            answers.append(dict(ans))
            ans['feedback'] = r'小行列式から余因子行列を構成する際，添字が逆転（転置）するので注意してください。また，行列式の扱いにも注意しましょう。'
            ans['data'] = [-_det, _incorrect_ans]
            answers.append(dict(ans))
            if abs(_det) > 1:
                ans['feedback'] = r'小行列式から余因子行列を構成する際，添字が逆転（転置）するので注意してください。また，行列式の扱いにも注意しましょう。'
                ans['data'] = [1/_det, _incorrect_ans]
                answers.append(dict(ans))
                ans['feedback'] = r'小行列式から余因子行列を構成する際，添字が逆転（転置）するので注意してください。また，行列式の扱いにも注意しましょう。'
                ans['data'] = [-1/_det, _incorrect_ans]
                answers.append(dict(ans))      
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_dim, _matrix, _adjugate, _det] = quiz.data
        _text = r'次の行列の逆行列を選んでください（余因子行列を活用してください）。<br />\( A='
        _text += sympy.latex(sympy.Matrix(_matrix), mat_delim='', mat_str='pmatrix') + r' \)<br />'
        _text += r'ただし，次の小行列式の情報を活用しても構いません。<br />\( '
        for _i in range(_dim):
            for _j in range(_dim):
                _text += r'|A_{' + str(_i+1) + r',' + str(_j+1) + r'}|=' + sympy.latex(sympy.Matrix(self._make_minor(_matrix,_i,_j)).det()) + r',\;'
        _text = _text[:-3]
        _text += r' \)'
        return _text
    def answer_text(self, ans):
        [_det, _adjugate] = ans['data']
        _text = r'\( '
        if _det == 1:
            _text += sympy.latex(sympy.Matrix(_adjugate), mat_delim='', mat_str='pmatrix')
        elif _det == -1:
            _text += r'-' + sympy.latex(sympy.Matrix(_adjugate), mat_delim='', mat_str='pmatrix')
        else:
            _text += sympy.latex(1/_det) + sympy.latex(sympy.Matrix(_adjugate), mat_delim='', mat_str='pmatrix')
        return _text + r' \)'


# In[4]:


if __name__ == "__main__":
    q = inverse_by_adjugate_matrix_with_minors()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=25)


# In[5]:


if __name__ == "__main__":
    pass
    #qz.save('inverse_by_adjugate_matrix_with_minors.xml')


# ## linear equation by Cramer rule

# In[193]:


class linear_equation_by_Cramer(core.Question):
    name = '線形方程式の解の計算（クラーメルの方法）'
    def __init__(self, dmin=3, dmax=3, emin=-3, emax=3):
        # 行列のサイズの範囲
        self.dim_min = dmin
        self.dim_max = dmax
        # 各要素の範囲
        self.elem_min = emin
        self.elem_max = emax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='クラーメルで線形方程式を解く', quiz_number=_quiz_number)
        _dim = random.randint(self.dim_min, self.dim_max)
        while True:
            _matrix = [[random.randint(self.elem_min,self.elem_max) for _j in range(_dim)] for _i in range(_dim)]
            _solution = [random.randint(self.elem_min,self.elem_max) for _j in range(_dim)]
            _constant = linear_algebra_next_generation.flatten_list_all((sympy.Matrix(_matrix)*sympy.Matrix([[_e] for _e in _solution])).tolist())
            if abs(sympy.Matrix(_matrix).det()) > 1 and functools.reduce(lambda x,y:abs(x)+abs(y), _solution) > 0:
                break
        quiz.quiz_identifier = hash(str(_matrix) + str(_constant) + str(_solution))
        # 正答の選択肢の生成
        ans = { 'fraction': 100, 'data': _solution }
        quiz.answers.append(ans)
        quiz.data = [_dim, _matrix, _constant, _solution]
        return quiz
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=5):
        answers = []
        ans = { 'fraction': 0 }
        [_dim, _matrix, _constant, _solution] = quiz.data
        _det = sympy.Matrix(_matrix).det()
        _rdets = [self._replace_part_matrix(_matrix, _constant, _r, True).det() for _r in range(_dim)]
        _cdets = [self._replace_part_matrix(_matrix, _constant, _c, False).det() for _c in range(_dim)]
        if _solution != _rdets:
            ans['feedback'] = r'クラーメルの公式で，係数行列の行列式をどのように用いているかを確認しましょう。'
            ans['data'] = _rdets
            answers.append(dict(ans))
        if _solution != _cdets:
            ans['feedback'] = r'クラーメルの公式をきちんと確認しましょう。'
            ans['data'] = _cdets
            answers.append(dict(ans))
        _incorrect_sol = [_e/_det for _e in _cdets]
        if _solution != _incorrect_sol:
            ans['feedback'] = r'クラーメルの公式で，分子の行列式の行列の構成方法を確認しましょう。'
            ans['data'] = _incorrect_sol
            answers.append(dict(ans))
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers
    def _replace_part_matrix(self, _matrix, _sol, _rowcol, _is_col):
        _smatrix = sympy.Matrix(_matrix)
        if _is_col:
            _smatrix[:,_rowcol] = _sol
        else:
            for _i in range(_smatrix.cols):
                _smatrix[_rowcol,_i] = _sol[_i]
        return _smatrix
    def question_text(self, quiz):
        [_dim, _matrix, _constant, _solution] = quiz.data
        _text = r'次の線形方程式の解をクラーメルの方法で求めよ。<br />\( \left\{\begin{array}{l}'
        _vars = [sympy.Symbol('x' + str(_i+1)) for _i in range(_dim)]
        for _i in range(_dim):
            _eq = 0
            for _j in range(_dim):
                _eq += _matrix[_i][_j] * _vars[_j]
            _text += sympy.latex(sympy.Eq(_eq, _constant[_i])) + r'\\'
        _text += r'\end{array}\right. \)<br />'
        _text += r'ただし，次の行列式の情報を活用しても構いません。<br />\( '
        _pairs = []
        for _r in range(_dim):
            _pairs.append([_r,True])
        for _c in range(_dim):
            _pairs.append([_c,False])
        _pairs = random.sample(_pairs, k=len(_pairs))
        for _p in _pairs:
            _smatrix = self._replace_part_matrix(_matrix, _constant, _p[0], _p[1])
            _text += sympy.latex(_smatrix, mat_delim='', mat_str='vmatrix')
            _text += r'=' + sympy.latex(_smatrix.det()) + r',\;'
        _text += sympy.latex(sympy.Matrix(_matrix), mat_delim='', mat_str='vmatrix')
        _text += r'=' + sympy.latex(sympy.Matrix(_matrix).det())
        _text += r' \)'
        return _text
    def answer_text(self, ans):
        _text = r'\( \left\{\begin{array}{l}'
        for _i in range(len(ans['data'])):
            _text += r'x_{' + str(_i+1) + r'}=' + sympy.latex(ans['data'][_i]) + r'\\'
        return _text + r'\end{array}\right. \)'


# In[194]:


if __name__ == "__main__":
    q = linear_equation_by_Cramer()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=25)


# In[195]:


if __name__ == "__main__":
    pass
    #qz.save('linear_equation_by_Cramer.xml')


# ## dummy

# In[ ]:





# # All the questions

# In[ ]:


questions_str = ['singular_or_nonsingular_ref', 'singular_or_nonsingular_nearly_ref', 'inverse_matrix_with_rref',
                 'inverse_matrix', 'composition_permutations', 'inverse_permutations', 'cycle_form_permutations', 
                 'transpositions_sign_permutations', 'determinant_by_sarrus', 'determinant_by_sarrus_plus_alpha', 
                 'same_determinant_by_row_ops', 'determinant_of_nearly_ref', 'minor_expansion', 'determinant_of_general_matrix', 
                 'adjugate_matrix_with_minors', 'inverse_by_adjugate_matrix_with_minors', 'linear_equation_by_Cramer']
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




