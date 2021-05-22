#!/usr/bin/env python
# coding: utf-8

# Moodle XML Question Generator, (C) 2019 Kosaku Nagasaka (Kobe University)

# ## モジュールファイルの生成

# In[ ]:


if __name__ == "__main__":
    import subprocess
    subprocess.run(['jupyter', 'nbconvert', '--to', 'python', '_linear_algebra_4.ipynb','--output','linear_algebra_4.py'])


# # Linear Algebra 4 (subspace and linear mapping)

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
    core._force_to_this_lang = 'ja'
else:
    from .. import core
    from . import common
    from . import linear_algebra


# In[2]:


import sympy
import random
import IPython
import itertools
import copy


# ## NOTES

# - Any matrix will be translated in the latex format with "pmatrix". Please translate it with your preferable environment (e.g. bmatrix) by the option of the top level function.
# - Any vector symbol will be translated in the latex format with "\vec". Please translate it with your preferable command (e.g. \boldsymbol) by the option of the top level function.

# ## select eigen value for the given eigen vector

# In[5]:


class eigenvalue_for_eigenvector(core.Question):
    name = '線形変換の固有ベクトルに対する固有値の選択'
    def __init__(self, emin=-3, emax=3, dmin=2, dmax=4, rmin=-3, rmax=3):
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 生成するベクトルの次元の範囲
        self.dim_min = dmin
        self.dim_max = dmax
        # 生成する固有値の範囲
        self.root_min = rmin
        self.root_max = rmax 
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='線形変換の固有値の選択', quiz_number=_quiz_number)
        es = linear_algebra.EigenSpace()
        es.set_dimension_range(self.dim_min, self.dim_max)
        es.set_element_range(self.elem_min, self.elem_max)
        es.set_root_range(self.root_min, self.root_max)
        es.generate(is_force_diagonalizable=None, is_force_symbolic=False, is_force_numeric=False)
        quiz.quiz_identifier = hash(es)
        # 正答の選択肢の生成
        _idx = random.choice(range(len(es.eigen_values)))
        quiz.data = [es, es.eigen_vectors[_idx][0]]
        ans = { 'fraction': 100, 'data': es.eigen_values[_idx]}
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        es = quiz.data[0]
        correct_data = quiz.data[1]
        for ev in range(self.root_min, self.root_max+1):
            if ev != correct_data:
                ans['feedback'] = '固有ベクトルと固有値の定義に立ち返り，固有ベクトルの像を計算すれば固有値は求められます。'
                ans['data'] = ev
                answers.append(dict(ans))                
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        es = quiz.data[0]
        vec = quiz.data[1]
        _text = r'次の線形変換を考えます。<br />'        
        _text += es.str_map(is_latex_closure=True)
        _text += r'<br />このとき，次のベクトルは，その固有ベクトルとなりますが，このときの固有値を選んでください。<br />'
        _text += es.str_vector(vec,is_latex_closure=True)
        return _text
    def answer_text(self, ans):
        _text = r'\(' + sympy.latex(ans['data']) + r'\)'
        return _text


# In[6]:


if __name__ == "__main__":
    q = eigenvalue_for_eigenvector()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=2)


# In[7]:


if __name__ == "__main__":
    pass
    #qz.save('eigenvalue_for_eigenvector.xml')


# ## select eigen vector

# In[21]:


class select_eigenvector(core.Question):
    name = '線形変換の固有ベクトルの選択'
    def __init__(self, emin=-3, emax=3, dmin=2, dmax=4, rmin=-3, rmax=3):
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 生成するベクトルの次元の範囲
        self.dim_min = dmin
        self.dim_max = dmax
        # 生成する固有値の範囲
        self.root_min = rmin
        self.root_max = rmax 
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='線形変換の固有ベクトルの選択', quiz_number=_quiz_number)
        es = linear_algebra.EigenSpace()
        es.set_dimension_range(self.dim_min, self.dim_max)
        es.set_element_range(self.elem_min, self.elem_max)
        es.set_root_range(self.root_min, self.root_max)
        es.generate(is_force_diagonalizable=True, is_force_symbolic=False, is_force_numeric=False)
        quiz.quiz_identifier = hash(es)
        # 正答の選択肢の生成
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
        # 正答を個別には作らないので何もしない
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
                ans['feedback'] = '固有ベクトルの定義に立ち返り，ベクトルの像を計算すれば確認できます。'
                ans['data'] = es.str_vector(_mat, is_latex_closure=True)
                answers.append(dict(ans))                
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        es = quiz.data[0]
        vec = quiz.data[1]
        _text = r'次の線形変換の固有ベクトルを選択してください。<br />'        
        _text += es.str_map(is_latex_closure=True)
        return _text
    def answer_text(self, ans):
        return ans['data']


# In[19]:


if __name__ == "__main__":
    q = select_eigenvector()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=2)


# In[20]:


if __name__ == "__main__":
    pass
    #qz.save('select_eigenvector.xml')


# ## characteristic polynomial

# In[51]:


class characteristic_polynomial(core.Question):
    name = '数行列の固有多項式の選択'
    def __init__(self, emin=-3, emax=3, dmin=2, dmax=3, rmin=-3, rmax=3, var=None):
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 生成するベクトルの次元の範囲
        self.dim_min = dmin
        self.dim_max = dmax
        # 生成する固有値の範囲
        self.root_min = rmin
        self.root_max = rmax
        if var is None:
            self.variable = sympy.Symbol('t')
        else:
            self.variable = var
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='固有多項式の選択', quiz_number=_quiz_number)
        es = linear_algebra.EigenSpace()
        es.set_dimension_range(self.dim_min, self.dim_max)
        es.set_element_range(self.elem_min, self.elem_max)
        es.set_root_range(self.root_min, self.root_max)
        es.generate(is_force_diagonalizable=True, is_force_symbolic=False, is_force_numeric=False)
        quiz.quiz_identifier = hash(es)
        # 正答の選択肢の生成
        quiz.data = [es]
        ans = { 'fraction': 100, 'data': es.characteristic_polynomial(self.variable)}
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        es = quiz.data[0]
        correct_data = es.characteristic_polynomial(self.variable)
        incorrect_data = sympy.expand(-1*correct_data)
        if sympy.expand(correct_data - incorrect_data) != 0:        
            ans['feedback'] = '定義における与えられた行列と単位行列の変数倍の順序を確認してください。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))
        _diagonals = [es.representation_matrix[i][i] for i in range(es.dimension)]
        incorrect_data = sympy.expand((self.variable*sympy.eye(es.dimension)-sympy.diag(*_diagonals)).det())
        if sympy.expand(correct_data - incorrect_data) != 0:        
            ans['feedback'] = '定義を確認しましょう。行列式をきちんと計算する必要があります。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))
        incorrect_data = sympy.expand(-1*incorrect_data)
        if sympy.expand(correct_data - incorrect_data) != 0:        
            ans['feedback'] = '定義を確認しましょう。行列式をきちんと計算する必要があります。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))
        _reverse_eye = [[0 if i != es.dimension - j - 1 else 1 for j in range(es.dimension)] for i in range(es.dimension)]
        incorrect_data = sympy.expand((self.variable*sympy.Matrix(_reverse_eye)-sympy.Matrix(es.representation_matrix)).det())
        if sympy.expand(correct_data - incorrect_data) != 0:        
            ans['feedback'] = '定義を確認しましょう。行列式をきちんと計算する必要があります。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))
        incorrect_data = sympy.expand(-1*incorrect_data)
        if sympy.expand(correct_data - incorrect_data) != 0:        
            ans['feedback'] = '定義を確認しましょう。行列式をきちんと計算する必要があります。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))
        answers = common.answer_union(answers)
        while len(answers) < size:
            _mat = sympy.Matrix(es.representation_matrix)
            _idxi = random.choice(range(es.dimension))
            _idxj = random.choice(range(es.dimension))
            _mat[_idxi,_idxj] += random.choice([-1,1])
            incorrect_data = sympy.expand((self.variable*sympy.eye(es.dimension)-_mat).det())
            if sympy.expand(correct_data - incorrect_data) != 0:        
                ans['feedback'] = '定義を確認しましょう。行列式をきちんと計算する必要があります。'
                ans['data'] = incorrect_data
                answers.append(dict(ans))
            incorrect_data = sympy.expand(-1*incorrect_data)
            if sympy.expand(correct_data - incorrect_data) != 0:        
                ans['feedback'] = '定義を確認しましょう。行列式をきちんと計算する必要があります。'
                ans['data'] = incorrect_data
                answers.append(dict(ans))
            answers = common.answer_union(answers)            
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        es = quiz.data[0]
        _text = r'次の行列の固有多項式を選んでください。<br />'        
        _text += es.str_matrix(is_latex_closure=True)
        return _text
    def answer_text(self, ans):
        return r'\(' + sympy.latex(ans['data']) + r'\)'


# In[52]:


if __name__ == "__main__":
    q = characteristic_polynomial()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=2)


# In[53]:


if __name__ == "__main__":
    pass
    #qz.save('characteristic_polynomial.xml')


# ## eigen space for numerical vector space

# In[52]:


class eigen_space_for_numeric_vector_space(core.Question):
    name = '数ベクトル空間の線形変換の固有空間の計算'
    def __init__(self, emin=-3, emax=3, dmin=2, dmax=3, rmin=-3, rmax=3):
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 生成するベクトルの次元の範囲
        self.dim_min = dmin
        self.dim_max = dmax
        # 生成する固有値の範囲
        self.root_min = rmin
        self.root_max = rmax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='固有空間の計算（数ベクトル空間）', quiz_number=_quiz_number)
        es = linear_algebra.EigenSpace()
        es.set_dimension_range(self.dim_min, self.dim_max)
        es.set_element_range(self.elem_min, self.elem_max)
        es.set_root_range(self.root_min, self.root_max)
        es.generate(is_force_diagonalizable=None, is_force_symbolic=False, is_force_numeric=True)
        quiz.quiz_identifier = hash(es)
        # 正答の選択肢の生成
        quiz.data = [es]
        ans = { 'fraction': 100, 'data': es.str_eigen_spaces(is_latex_closure=True)}
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        es = quiz.data[0]
        if len(es.eigen_values) > 1:
            incorrect_ev = es.eigen_values[1:] + es.eigen_values[:1]
            if not es.is_an_eigen_vector_for_eigen_value(es.eigen_vectors[0][0], es.eigen_values[1]):
                ans['feedback'] = '固有値と固有ベクトルは対になります。取り違いをしている可能性があります。'
                ans['data'] = es.str_eigen_spaces_for_the_given(incorrect_ev, es.eigen_vectors, es.diagonalizable, es.domain)
                answers.append(dict(ans))
        if max(es.eigen_space_dimensions) > 1:
            incorrect_ev = linear_algebra.flatten_list([[es.eigen_values[i] for j in range(es.eigen_space_dimensions[i])] for i in range(len(es.eigen_space_dimensions))])
            ans['feedback'] = '1つの固有値に対して，1つの空間が定まります。'
            ans['data'] = es.str_eigen_spaces_for_the_given(incorrect_ev, [[e] for e in linear_algebra.flatten_list(es.eigen_vectors)], es.diagonalizable, es.domain)
            answers.append(dict(ans))
            if len(es.eigen_values) > 1:
                incorrect_ev = incorrect_ev[1:] + incorrect_ev[:1]
                ans['feedback'] = '1つの固有値に対して，1つの空間が定まります。'
                ans['data'] = es.str_eigen_spaces_for_the_given(incorrect_ev, [[e] for e in linear_algebra.flatten_list(es.eigen_vectors)], es.diagonalizable, es.domain)
                answers.append(dict(ans))
        answers = common.answer_union(answers)
        while len(answers) < size:
            num_evs = len(es.eigen_values)
            incorrect_ev = random.sample(list(set(range(-2*num_evs,2*num_evs+1))-set(es.eigen_values)), num_evs)
            ans['feedback'] = 'まずは，固有値を正しく求めましょう。'
            ans['data'] = es.str_eigen_spaces_for_the_given(incorrect_ev, es.eigen_vectors, es.diagonalizable, es.domain)
            answers.append(dict(ans))
            answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        es = quiz.data[0]
        _text = r'次の線形変換の各固有値の固有空間（標準基底に関する表現行列で）を求めてください。<br />'        
        _text += es.str_map(is_latex_closure=True)
        return _text
    def answer_text(self, ans):
        return ans['data']


# In[53]:


if __name__ == "__main__":
    q = eigen_space_for_numeric_vector_space()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[54]:


if __name__ == "__main__":
    pass
    #qz.save('eigen_space_for_numeric_vector_space.xml')


# ## eigen space for polynomial vector space

# In[55]:


class eigen_space_for_polynomial_vector_space(core.Question):
    name = 'ベクトル空間（多項式）の線形変換の固有空間の計算'
    def __init__(self, emin=-3, emax=3, dmin=2, dmax=3, rmin=-3, rmax=3):
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 生成するベクトルの次元の範囲
        self.dim_min = dmin
        self.dim_max = dmax
        # 生成する固有値の範囲
        self.root_min = rmin
        self.root_max = rmax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='固有空間の計算（多項式）', quiz_number=_quiz_number)
        es = linear_algebra.EigenSpace()
        es.set_dimension_range(self.dim_min, self.dim_max)
        es.set_element_range(self.elem_min, self.elem_max)
        es.set_root_range(self.root_min, self.root_max)
        es.generate(is_force_diagonalizable=None, is_force_symbolic=False, is_force_polynomial=True)
        quiz.quiz_identifier = hash(es)
        # 正答の選択肢の生成
        quiz.data = [es]
        ans = { 'fraction': 100, 'data': es.str_eigen_spaces(is_latex_closure=True)}
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        es = quiz.data[0]
        if len(es.eigen_values) > 1:
            incorrect_ev = es.eigen_values[1:] + es.eigen_values[:1]
            if not es.is_an_eigen_vector_for_eigen_value(es.eigen_vectors[0][0], es.eigen_values[1]):
                ans['feedback'] = '固有値と固有ベクトルは対になります。取り違いをしている可能性があります。'
                ans['data'] = es.str_eigen_spaces_for_the_given(incorrect_ev, es.eigen_vectors, es.diagonalizable, es.domain)
                answers.append(dict(ans))
        if max(es.eigen_space_dimensions) > 1:
            incorrect_ev = linear_algebra.flatten_list([[es.eigen_values[i] for j in range(es.eigen_space_dimensions[i])] for i in range(len(es.eigen_space_dimensions))])
            ans['feedback'] = '1つの固有値に対して，1つの空間が定まります。'
            ans['data'] = es.str_eigen_spaces_for_the_given(incorrect_ev, [[e] for e in linear_algebra.flatten_list(es.eigen_vectors)], es.diagonalizable, es.domain)
            answers.append(dict(ans))
            if len(es.eigen_values) > 1:
                incorrect_ev = incorrect_ev[1:] + incorrect_ev[:1]
                ans['feedback'] = '1つの固有値に対して，1つの空間が定まります。'
                ans['data'] = es.str_eigen_spaces_for_the_given(incorrect_ev, [[e] for e in linear_algebra.flatten_list(es.eigen_vectors)], es.diagonalizable, es.domain)
                answers.append(dict(ans))
        answers = common.answer_union(answers)
        while len(answers) < size:
            num_evs = len(es.eigen_values)
            incorrect_ev = random.sample(list(set(range(-2*num_evs,2*num_evs+1))-set(es.eigen_values)), num_evs)
            ans['feedback'] = 'まずは，固有値を正しく求めましょう。'
            ans['data'] = es.str_eigen_spaces_for_the_given(incorrect_ev, es.eigen_vectors, es.diagonalizable, es.domain)
            answers.append(dict(ans))
            answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        es = quiz.data[0]
        _text = r'次の線形変換の各固有値の固有空間（昇冪の基底\(\{'
        _x = sympy.Symbol('x')
        for i in range(es.dimension):
            _text += sympy.latex(_x**i)
            if i < es.dimension - 1:
                _text += r','
        _text += r'\}\)に関する表現行列で）を求めてください。<br />'   
        _text += es.str_map(is_latex_closure=True)
        return _text
    def answer_text(self, ans):
        return ans['data']


# In[56]:


if __name__ == "__main__":
    q = eigen_space_for_polynomial_vector_space()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[57]:


if __name__ == "__main__":
    pass
    #qz.save('eigen_space_for_polynomial_vector_space.xml')


# ## select diagonalizable representation matrix

# In[8]:


class select_diagonalizable_representation_matrix(core.Question):
    name = '表現行列が対角化可能な線形変換の選択'
    def __init__(self, emin=-3, emax=3, dmin=2, dmax=4, rmin=-3, rmax=3):
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 生成するベクトルの次元の範囲
        self.dim_min = dmin
        self.dim_max = dmax
        # 生成する固有値の範囲
        self.root_min = rmin
        self.root_max = rmax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='対角化可能な線形変換の選択', quiz_number=_quiz_number)
        es = linear_algebra.EigenSpace()
        es.set_dimension_range(self.dim_min, self.dim_max)
        es.set_element_range(self.elem_min, self.elem_max)
        es.set_root_range(self.root_min, self.root_max)
        es.generate(is_force_diagonalizable=True, is_force_symbolic=False, is_force_numeric=False, is_force_polynomial=False)
        quiz.quiz_identifier = hash(es)
        # 正答の選択肢の生成
        quiz.data = [es]
        ans = { 'fraction': 100, 'data': es }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        es = quiz.data[0]
        while len(answers) < size:
            new_es = linear_algebra.EigenSpace(es)
            new_es.generate(is_force_diagonalizable=False, is_force_symbolic=False, is_force_numeric=False, is_force_polynomial=False)
            ans['feedback'] = '対角化可能な条件を固有空間の次元が満たしている丁寧に確認してください。'
            ans['data'] = new_es
            answers.append(dict(ans))
            answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        _text = r'次の線形変換の中で，その固有空間を参考にして，表現表列が対角化可能なものを選択してください。'        
        return _text
    def answer_text(self, ans):
        es = ans['data']
        _text = es.str_map(is_latex_closure=True) + r'<br />'
        _text += es.str_eigen_spaces(is_latex_closure=True)
        return _text


# In[10]:


if __name__ == "__main__":
    q = select_diagonalizable_representation_matrix()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[11]:


if __name__ == "__main__":
    pass
    #qz.save('select_diagonalizable_representation_matrix.xml')


# ## diagonalization from numeric eigen spaces

# In[28]:


class diagonalization_from_numeric_eigen_spaces(core.Question):
    name = '数ベクトル空間における線形変換の表現行列の対角化（固有空間から）'
    def __init__(self, emin=-3, emax=3, dmin=2, dmax=4, rmin=-3, rmax=3):
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 生成するベクトルの次元の範囲
        self.dim_min = dmin
        self.dim_max = dmax
        # 生成する固有値の範囲
        self.root_min = rmin
        self.root_max = rmax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='表現行列の対角化（数ベクトル空間，固有空間から）', quiz_number=_quiz_number)
        es = linear_algebra.EigenSpace()
        es.set_dimension_range(self.dim_min, self.dim_max)
        es.set_element_range(self.elem_min, self.elem_max)
        es.set_root_range(self.root_min, self.root_max)
        es.generate(is_force_diagonalizable=True, is_force_symbolic=False, is_force_numeric=True, is_force_polynomial=False)
        quiz.quiz_identifier = hash(es)
        # 正答の選択肢の生成
        _evcts = linear_algebra.flatten_list(es.eigen_vectors)
        _evs = linear_algebra.flatten_list([[es.eigen_values[i] for j in range(len(es.eigen_vectors[i]))] for i in range(len(es.eigen_values))])
        quiz.data = [es, _evs, _evcts]
        ans = { 'fraction': 100, 'data': [_evs, _evcts] }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def _is_correct(self, matA, evs, evcts):
        _matP = sympy.Matrix(evcts).transpose()
        _matD = sympy.diag(*evs)
        if _matP.inv()*matA*_matP == _matD:
            return True
        else:
            return False
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        es = quiz.data[0]
        correct_evs = quiz.data[1]
        correct_evcts = quiz.data[2]
        matA = sympy.Matrix(es.representation_matrix)
        incorrect_evcts = sympy.matrix2numpy(sympy.Matrix(correct_evcts).transpose()).tolist()
        incorrect_evs = correct_evs
        if not self._is_correct(matA, incorrect_evs, incorrect_evcts):
            ans['feedback'] = '対角化に用いる正則な行列の列ベクトルが，固有ベクトルに対応します。'
            ans['data'] = [incorrect_evs, incorrect_evcts]
            answers.append(dict(ans))
        for i in range(len(correct_evs)):
            incorrect_evs = correct_evs[i:] + correct_evs[:i]
            incorrect_evcts = correct_evcts
            if not self._is_correct(matA, incorrect_evs, incorrect_evcts):
                ans['feedback'] = '対角化に用いる正則な行列の列ベクトルは，位置関係から対応する固有値の固有空間の固有ベクトルに対応します。'
                ans['data'] = [incorrect_evs, incorrect_evcts]
                answers.append(dict(ans))
            incorrect_evcts = sympy.matrix2numpy(sympy.Matrix(incorrect_evcts).transpose()).tolist()
            if not self._is_correct(matA, incorrect_evs, incorrect_evcts):
                ans['feedback'] = '対角化に用いる正則な行列の列ベクトルは，位置関係から対応する固有値の固有空間の固有ベクトルに対応します。'
                ans['data'] = [incorrect_evs, incorrect_evcts]
                answers.append(dict(ans))
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        es = quiz.data[0]
        _text = r'次の線形変換について，その固有空間を参考に，その表現表列\(A\)（標準基底に関する）を対角化してください。<br />'
        _text += es.str_map(is_latex_closure=True) + r'<br />'
        _text += es.str_eigen_spaces(is_latex_closure=True)
        return _text
    def answer_text(self, ans):
        evs = ans['data'][0]
        evcts = ans['data'][1]
        _matP = sympy.Matrix(evcts).transpose()
        _matD = sympy.diag(*evs)
        _text = r'\(P=' + sympy.latex(_matP, mat_delim='', mat_str='pmatrix') + r',\;'
        _text += r'P^{-1}AP=' + sympy.latex(_matD, mat_delim='', mat_str='pmatrix') + r'\)'
        return _text


# In[29]:


if __name__ == "__main__":
    q = diagonalization_from_numeric_eigen_spaces()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[30]:


if __name__ == "__main__":
    pass
    #qz.save('diagonalization_from_numeric_eigen_spaces.xml')


# ## diagonalization from polynomial eigen spaces

# In[37]:


class diagonalization_from_polynomial_eigen_spaces(core.Question):
    name = '多項式のベクトル空間における線形変換の表現行列の対角化（固有空間から）'
    def __init__(self, emin=-3, emax=3, dmin=2, dmax=4, rmin=-3, rmax=3):
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 生成するベクトルの次元の範囲
        self.dim_min = dmin
        self.dim_max = dmax
        # 生成する固有値の範囲
        self.root_min = rmin
        self.root_max = rmax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='表現行列の対角化（多項式，固有空間から）', quiz_number=_quiz_number)
        es = linear_algebra.EigenSpace()
        es.set_dimension_range(self.dim_min, self.dim_max)
        es.set_element_range(self.elem_min, self.elem_max)
        es.set_root_range(self.root_min, self.root_max)
        es.generate(is_force_diagonalizable=True, is_force_symbolic=False, is_force_numeric=False, is_force_polynomial=True)
        quiz.quiz_identifier = hash(es)
        # 正答の選択肢の生成
        _evcts = linear_algebra.flatten_list(es.eigen_vectors)
        _evs = linear_algebra.flatten_list([[es.eigen_values[i] for j in range(len(es.eigen_vectors[i]))] for i in range(len(es.eigen_values))])
        quiz.data = [es, _evs, _evcts]
        ans = { 'fraction': 100, 'data': [_evs, _evcts] }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def _is_correct(self, matA, evs, evcts):
        _matP = sympy.Matrix(evcts).transpose()
        _matD = sympy.diag(*evs)
        if _matP.inv()*matA*_matP == _matD:
            return True
        else:
            return False
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        es = quiz.data[0]
        correct_evs = quiz.data[1]
        correct_evcts = quiz.data[2]
        matA = sympy.Matrix(es.representation_matrix)
        incorrect_evcts = sympy.matrix2numpy(sympy.Matrix(correct_evcts).transpose()).tolist()
        incorrect_evs = correct_evs
        if not self._is_correct(matA, incorrect_evs, incorrect_evcts):
            ans['feedback'] = '対角化に用いる正則な行列の列ベクトルが，固有ベクトルに対応します。'
            ans['data'] = [incorrect_evs, incorrect_evcts]
            answers.append(dict(ans))
        for i in range(len(correct_evs)):
            incorrect_evs = correct_evs[i:] + correct_evs[:i]
            incorrect_evcts = correct_evcts
            if not self._is_correct(matA, incorrect_evs, incorrect_evcts):
                ans['feedback'] = '対角化に用いる正則な行列の列ベクトルは，位置関係から対応する固有値の固有空間の固有ベクトルに対応します。'
                ans['data'] = [incorrect_evs, incorrect_evcts]
                answers.append(dict(ans))
            incorrect_evcts = sympy.matrix2numpy(sympy.Matrix(incorrect_evcts).transpose()).tolist()
            if not self._is_correct(matA, incorrect_evs, incorrect_evcts):
                ans['feedback'] = '対角化に用いる正則な行列の列ベクトルは，位置関係から対応する固有値の固有空間の固有ベクトルに対応します。'
                ans['data'] = [incorrect_evs, incorrect_evcts]
                answers.append(dict(ans))
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        es = quiz.data[0]
        _text = r'次の線形変換について，その固有空間を参考に，その表現表列\(A\)（昇冪の基底\(\{'
        _x = sympy.Symbol('x')
        for i in range(es.dimension):
            _text += sympy.latex(_x**i)
            if i < es.dimension - 1:
                _text += r','
        _text += r'\}\)に関する）を対角化してください。<br />'
        _text += es.str_map(is_latex_closure=True) + r'<br />'
        _text += es.str_eigen_spaces(is_latex_closure=True)
        return _text
    def answer_text(self, ans):
        evs = ans['data'][0]
        evcts = ans['data'][1]
        _matP = sympy.Matrix(evcts).transpose()
        _matD = sympy.diag(*evs)
        _text = r'\(P=' + sympy.latex(_matP, mat_delim='', mat_str='pmatrix') + r',\;'
        _text += r'P^{-1}AP=' + sympy.latex(_matD, mat_delim='', mat_str='pmatrix') + r'\)'
        return _text


# In[38]:


if __name__ == "__main__":
    q = diagonalization_from_polynomial_eigen_spaces()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[39]:


if __name__ == "__main__":
    pass
    #qz.save('diagonalization_from_polynomial_eigen_spaces.xml')


# ## diagonalization of matrix

# In[40]:


class diagonalization_of_matrix(core.Question):
    name = '行列の対角化（ヒントなし）'
    def __init__(self, emin=-3, emax=3, dmin=2, dmax=3, rmin=-3, rmax=3):
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 生成するベクトルの次元の範囲
        self.dim_min = dmin
        self.dim_max = dmax
        # 生成する固有値の範囲
        self.root_min = rmin
        self.root_max = rmax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='行列の対角化', quiz_number=_quiz_number)
        es = linear_algebra.EigenSpace()
        es.set_dimension_range(self.dim_min, self.dim_max)
        es.set_element_range(self.elem_min, self.elem_max)
        es.set_root_range(self.root_min, self.root_max)
        es.generate(is_force_diagonalizable=True, is_force_symbolic=True, is_force_numeric=True, is_force_polynomial=False)
        quiz.quiz_identifier = hash(es)
        # 正答の選択肢の生成
        _evcts = linear_algebra.flatten_list(es.eigen_vectors)
        _evs = linear_algebra.flatten_list([[es.eigen_values[i] for j in range(len(es.eigen_vectors[i]))] for i in range(len(es.eigen_values))])
        quiz.data = [es, _evs, _evcts]
        ans = { 'fraction': 100, 'data': [_evs, _evcts] }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def _is_correct(self, matA, evs, evcts):
        _matP = sympy.Matrix(evcts).transpose()
        _matD = sympy.diag(*evs)
        if _matP.inv()*matA*_matP == _matD:
            return True
        else:
            return False
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        es = quiz.data[0]
        correct_evs = quiz.data[1]
        correct_evcts = quiz.data[2]
        matA = sympy.Matrix(es.representation_matrix)
        incorrect_evcts = sympy.matrix2numpy(sympy.Matrix(correct_evcts).transpose()).tolist()
        incorrect_evs = correct_evs
        if not self._is_correct(matA, incorrect_evs, incorrect_evcts):
            ans['feedback'] = '対角化に用いる正則な行列の列ベクトルが，固有ベクトルに対応します。'
            ans['data'] = [incorrect_evs, incorrect_evcts]
            answers.append(dict(ans))
        for i in range(len(correct_evs)):
            incorrect_evs = correct_evs[i:] + correct_evs[:i]
            incorrect_evcts = correct_evcts
            if not self._is_correct(matA, incorrect_evs, incorrect_evcts):
                ans['feedback'] = '対角化に用いる正則な行列の列ベクトルは，位置関係から対応する固有値の固有空間の固有ベクトルに対応します。'
                ans['data'] = [incorrect_evs, incorrect_evcts]
                answers.append(dict(ans))
            incorrect_evcts = sympy.matrix2numpy(sympy.Matrix(incorrect_evcts).transpose()).tolist()
            if not self._is_correct(matA, incorrect_evs, incorrect_evcts):
                ans['feedback'] = '対角化に用いる正則な行列の列ベクトルは，位置関係から対応する固有値の固有空間の固有ベクトルに対応します。'
                ans['data'] = [incorrect_evs, incorrect_evcts]
                answers.append(dict(ans))
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        es = quiz.data[0]
        _text = r'次の行列を対角化してください。<br />'
        _text += es.str_matrix(is_latex_closure=True)
        return _text
    def answer_text(self, ans):
        evs = ans['data'][0]
        evcts = ans['data'][1]
        _matP = sympy.Matrix(evcts).transpose()
        _matD = sympy.diag(*evs)
        _text = r'\(P=' + sympy.latex(_matP, mat_delim='', mat_str='pmatrix') + r',\;'
        _text += r'P^{-1}AP=' + sympy.latex(_matD, mat_delim='', mat_str='pmatrix') + r'\)'
        return _text


# In[41]:


if __name__ == "__main__":
    q = diagonalization_of_matrix()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[42]:


if __name__ == "__main__":
    pass
    #qz.save('diagonalization_of_matrix.xml')


# ## diagonalization of map

# In[49]:


class diagonalization_of_map(core.Question):
    name = '線形変換の表現行列の対角化（ヒントなし）'
    def __init__(self, emin=-3, emax=3, dmin=2, dmax=3, rmin=-3, rmax=3):
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 生成するベクトルの次元の範囲
        self.dim_min = dmin
        self.dim_max = dmax
        # 生成する固有値の範囲
        self.root_min = rmin
        self.root_max = rmax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='線形変換の表現行列の対角化', quiz_number=_quiz_number)
        es = linear_algebra.EigenSpace()
        es.set_dimension_range(self.dim_min, self.dim_max)
        es.set_element_range(self.elem_min, self.elem_max)
        es.set_root_range(self.root_min, self.root_max)
        es.generate(is_force_diagonalizable=True, is_force_symbolic=False, is_force_numeric=False, is_force_polynomial=False)
        quiz.quiz_identifier = hash(es)
        # 正答の選択肢の生成
        _evcts = linear_algebra.flatten_list(es.eigen_vectors)
        _evs = linear_algebra.flatten_list([[es.eigen_values[i] for j in range(len(es.eigen_vectors[i]))] for i in range(len(es.eigen_values))])
        quiz.data = [es, _evs, _evcts]
        ans = { 'fraction': 100, 'data': [_evs, _evcts] }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def _is_correct(self, matA, evs, evcts):
        _matP = sympy.Matrix(evcts).transpose()
        _matD = sympy.diag(*evs)
        if _matP.inv()*matA*_matP == _matD:
            return True
        else:
            return False
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        es = quiz.data[0]
        correct_evs = quiz.data[1]
        correct_evcts = quiz.data[2]
        matA = sympy.Matrix(es.representation_matrix)
        incorrect_evcts = sympy.matrix2numpy(sympy.Matrix(correct_evcts).transpose()).tolist()
        incorrect_evs = correct_evs
        if not self._is_correct(matA, incorrect_evs, incorrect_evcts):
            ans['feedback'] = '対角化に用いる正則な行列の列ベクトルが，固有ベクトルに対応します。'
            ans['data'] = [incorrect_evs, incorrect_evcts]
            answers.append(dict(ans))
        for i in range(len(correct_evs)):
            incorrect_evs = correct_evs[i:] + correct_evs[:i]
            incorrect_evcts = correct_evcts
            if not self._is_correct(matA, incorrect_evs, incorrect_evcts):
                ans['feedback'] = '対角化に用いる正則な行列の列ベクトルは，位置関係から対応する固有値の固有空間の固有ベクトルに対応します。'
                ans['data'] = [incorrect_evs, incorrect_evcts]
                answers.append(dict(ans))
            incorrect_evcts = sympy.matrix2numpy(sympy.Matrix(incorrect_evcts).transpose()).tolist()
            if not self._is_correct(matA, incorrect_evs, incorrect_evcts):
                ans['feedback'] = '対角化に用いる正則な行列の列ベクトルは，位置関係から対応する固有値の固有空間の固有ベクトルに対応します。'
                ans['data'] = [incorrect_evs, incorrect_evcts]
                answers.append(dict(ans))
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        es = quiz.data[0]
        _text = r'次の線形変換について，その表現表列\(A\)（'
        if es.domain == 'numeric':
            _text += r'標準基底に関する'
        else: # polynomial
            _text += r'昇冪の基底\(\{'
            _x = sympy.Symbol('x')
            for i in range(es.dimension):
                _text += sympy.latex(_x**i)
                if i < es.dimension - 1:
                    _text += r','
            _text += r'\}\)に関する'     
        _text += r'）を対角化してください。<br />'
        _text += es.str_map(is_latex_closure=True)
        return _text
    def answer_text(self, ans):
        evs = ans['data'][0]
        evcts = ans['data'][1]
        _matP = sympy.Matrix(evcts).transpose()
        _matD = sympy.diag(*evs)
        _text = r'\(P=' + sympy.latex(_matP, mat_delim='', mat_str='pmatrix') + r',\;'
        _text += r'P^{-1}AP=' + sympy.latex(_matD, mat_delim='', mat_str='pmatrix') + r'\)'
        return _text


# In[50]:


if __name__ == "__main__":
    q = diagonalization_of_map()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[51]:


if __name__ == "__main__":
    pass
    #qz.save('diagonalization_of_map.xml')


# ## standard inner product and norm in Rn

# In[10]:


class standard_inner_product_with_norm_in_Rn(core.Question):
    name = '標準内積とそのノルム（数ベクトル空間）'
    def __init__(self, emin=-3, emax=3, dmin=2, dmax=3):
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 生成するベクトルの次元の範囲
        self.dim_min = dmin
        self.dim_max = dmax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='標準内積とそのノルム', quiz_number=_quiz_number)
        ins = linear_algebra.InnerSpace(domain='Rn')
        _dimension = random.randint(self.dim_min, self.dim_max)
        _vectA = [random.randint(self.elem_min, self.elem_max) for i in range(_dimension)]
        _vectB = [random.randint(self.elem_min, self.elem_max) for i in range(_dimension)]
        quiz.quiz_identifier = hash(str(_vectA) + str(_vectB))
        # 正答の選択肢の生成
        _inner_product = ins.inner_product(_vectA, _vectB)
        _normA = ins.norm(_vectA)
        _normB = ins.norm(_vectB)
        quiz.data = [_dimension, _vectA, _vectB, _inner_product, _normA, _normB]
        ans = { 'fraction': 100, 'data': [_inner_product, _normA, _normB] }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_dimension, _vectA, _vectB, _inner_product, _normA, _normB] = quiz.data
        ins = linear_algebra.InnerSpace(domain='Rn')
        ans['feedback'] = '標準内積は，数ベクトル空間における主たる内積です。定義されています。'
        ans['data'] = [r'標準内積はこのベクトル空間では定義されない。']
        answers.append(dict(ans))
        incorrect_inner_product = -_inner_product
        incorrect_normA = _normA
        incorrect_normB = _normB
        if incorrect_inner_product != _inner_product:
            ans['feedback'] = '標準内積は，ベクトルの各要素の積の和です。計算し直しましょう。'
            ans['data'] = [incorrect_inner_product, incorrect_normA, incorrect_normB]
            answers.append(dict(ans))
        incorrect_inner_product = _inner_product
        incorrect_normA = ins.inner_product(_vectA, _vectA)
        incorrect_normB = ins.inner_product(_vectB, _vectB)
        if incorrect_inner_product != _inner_product or incorrect_normA != _normA or incorrect_normB != _normB:
            ans['feedback'] = 'ノルムは，自分自身との内積の二乗根です。計算し直しましょう。'
            ans['data'] = [incorrect_inner_product, incorrect_normA, incorrect_normB]
            answers.append(dict(ans))
        incorrect_inner_product = -_inner_product
        if incorrect_inner_product != _inner_product or incorrect_normA != _normA or incorrect_normB != _normB:
            ans['feedback'] = 'ノルムは，自分自身との内積の二乗根です。計算し直しましょう。'
            ans['data'] = [incorrect_inner_product, incorrect_normA, incorrect_normB]
            answers.append(dict(ans))
        incorrect_inner_product = sum(_vectA) + sum(_vectB)
        incorrect_normA = abs(sum(_vectA))
        incorrect_normB = abs(sum(_vectB))
        if incorrect_inner_product != _inner_product or incorrect_normA != _normA or incorrect_normB != _normB:
            ans['feedback'] = '標準内積は，ベクトルの各要素の積の和です。計算し直しましょう。'
            ans['data'] = [incorrect_inner_product, incorrect_normA, incorrect_normB]
            answers.append(dict(ans))
        incorrect_inner_product = -_inner_product
        if incorrect_inner_product != _inner_product or incorrect_normA != _normA or incorrect_normB != _normB:
            ans['feedback'] = '標準内積は，ベクトルの各要素の積の和です。計算し直しましょう。'
            ans['data'] = [incorrect_inner_product, incorrect_normA, incorrect_normB]
            answers.append(dict(ans))
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_dimension, _vectA, _vectB, _inner_product, _normA, _normB] = quiz.data
        ins = linear_algebra.InnerSpace(domain='Rn')
        _text = r'ベクトル空間' + ins.str_domain(is_latex_closure=True) + r'の内積を標準内積とする。'
        _text += r'このとき，次の2つのベクトルの内積とそれぞれのノルムとして適切なものを選択してください。<br />'
        _text += ins.str_vectors([_vectA, _vectB])
        return _text
    def answer_text(self, ans):
        if len(ans['data']) > 1:
            [_inner_product, _normA, _normB] = ans['data']
            _text = r'内積: \(' + sympy.latex(_inner_product) + r'\), '
            _text += r'ノルム: \(' + sympy.latex(_normA) + r',\;' + sympy.latex(_normB) + r'\)'
        else:
            _text = ans['data'][0]
        return _text


# In[11]:


if __name__ == "__main__":
    q = standard_inner_product_with_norm_in_Rn()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[12]:


if __name__ == "__main__":
    pass
    #qz.save('standard_inner_product_with_norm_in_Rn.xml')


# ## another inner product and norm in Rn

# In[13]:


class another_inner_product_with_norm_in_Rn(core.Question):
    name = '標準内積ではない内積とそのノルム（数ベクトル空間）'
    def __init__(self, emin=-3, emax=3, dmin=2, dmax=3):
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 生成するベクトルの次元の範囲
        self.dim_min = dmin
        self.dim_max = dmax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='他の内積とそのノルム', quiz_number=_quiz_number)
        ins = linear_algebra.InnerSpace(domain='aRn')
        _dimension = random.randint(self.dim_min, self.dim_max)
        _vectA = [random.randint(self.elem_min, self.elem_max) for i in range(_dimension)]
        _vectB = [random.randint(self.elem_min, self.elem_max) for i in range(_dimension)]
        quiz.quiz_identifier = hash(str(_vectA) + str(_vectB))
        # 正答の選択肢の生成
        _inner_product = ins.inner_product(_vectA, _vectB)
        _normA = ins.norm(_vectA)
        _normB = ins.norm(_vectB)
        quiz.data = [_dimension, _vectA, _vectB, _inner_product, _normA, _normB]
        ans = { 'fraction': 100, 'data': [_inner_product, _normA, _normB] }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_dimension, _vectA, _vectB, _inner_product, _normA, _normB] = quiz.data
        ins = linear_algebra.InnerSpace(domain='aRn')
        ans['feedback'] = '今回の内積は，きちんと内積の条件を満たしています。'
        ans['data'] = [r'この内積はこのベクトル空間では内積の条件を満たさず，定義されない。']
        answers.append(dict(ans))
        ins_standard = linear_algebra.InnerSpace(domain='Rn')
        incorrect_inner_product = ins_standard.inner_product(_vectA, _vectB)
        incorrect_normA = ins_standard.norm(_vectA)
        incorrect_normB = ins_standard.norm(_vectB)
        if incorrect_inner_product != _inner_product or incorrect_normA != _normA or incorrect_normB != _normB:
            ans['feedback'] = '標準内積ではありません。与えられた内積の定義に基づいて計算し直しましょう。'
            ans['data'] = [incorrect_inner_product, incorrect_normA, incorrect_normB]
            answers.append(dict(ans))
        incorrect_inner_product = -_inner_product
        if incorrect_inner_product != _inner_product or incorrect_normA != _normA or incorrect_normB != _normB:
            ans['feedback'] = '標準内積ではありません。与えられた内積の定義に基づいて計算し直しましょう。'
            ans['data'] = [incorrect_inner_product, incorrect_normA, incorrect_normB]
            answers.append(dict(ans))
        incorrect_inner_product = -_inner_product
        incorrect_normA = _normA
        incorrect_normB = _normB
        if incorrect_inner_product != _inner_product:
            ans['feedback'] = '与えられた内積の定義に基づいて計算し直しましょう。'
            ans['data'] = [incorrect_inner_product, incorrect_normA, incorrect_normB]
            answers.append(dict(ans))
        incorrect_inner_product = _inner_product
        incorrect_normA = ins.inner_product(_vectA, _vectA)
        incorrect_normB = ins.inner_product(_vectB, _vectB)
        if incorrect_inner_product != _inner_product or incorrect_normA != _normA or incorrect_normB != _normB:
            ans['feedback'] = 'ノルムは，自分自身との内積の二乗根です。計算し直しましょう。'
            ans['data'] = [incorrect_inner_product, incorrect_normA, incorrect_normB]
            answers.append(dict(ans))
        incorrect_inner_product = -_inner_product
        if incorrect_inner_product != _inner_product or incorrect_normA != _normA or incorrect_normB != _normB:
            ans['feedback'] = 'ノルムは，自分自身との内積の二乗根です。計算し直しましょう。'
            ans['data'] = [incorrect_inner_product, incorrect_normA, incorrect_normB]
            answers.append(dict(ans))
        incorrect_inner_product = sum(_vectA) + sum(_vectB)
        incorrect_normA = abs(sum(_vectA))
        incorrect_normB = abs(sum(_vectB))
        if incorrect_inner_product != _inner_product or incorrect_normA != _normA or incorrect_normB != _normB:
            ans['feedback'] = '与えられた内積の定義に基づいて計算し直しましょう。'
            ans['data'] = [incorrect_inner_product, incorrect_normA, incorrect_normB]
            answers.append(dict(ans))
        incorrect_inner_product = -_inner_product
        if incorrect_inner_product != _inner_product or incorrect_normA != _normA or incorrect_normB != _normB:
            ans['feedback'] = '与えられた内積の定義に基づいて計算し直しましょう。'
            ans['data'] = [incorrect_inner_product, incorrect_normA, incorrect_normB]
            answers.append(dict(ans))
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_dimension, _vectA, _vectB, _inner_product, _normA, _normB] = quiz.data
        ins = linear_algebra.InnerSpace(domain='aRn')
        _text = r'ベクトル空間' + ins.str_domain(is_latex_closure=True) + r'の内積を'
        _text += ins.str_inner_product(is_latex_closure=True)
        _text += r'とする。このとき，次の2つのベクトルの内積とそれぞれのノルムとして適切なものを選択してください。<br />'
        _text += ins.str_vectors([_vectA, _vectB])
        return _text
    def answer_text(self, ans):
        if len(ans['data']) > 1:
            [_inner_product, _normA, _normB] = ans['data']
            _text = r'内積: \(' + sympy.latex(_inner_product) + r'\), '
            _text += r'ノルム: \(' + sympy.latex(_normA) + r',\;' + sympy.latex(_normB) + r'\)'
        else:
            _text = ans['data'][0]
        return _text


# In[14]:


if __name__ == "__main__":
    q = another_inner_product_with_norm_in_Rn()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[15]:


if __name__ == "__main__":
    pass
    #qz.save('another_inner_product_with_norm_in_Rn.xml')


# ## some inner product and norm in Rnm

# In[16]:


class some_inner_product_with_norm_in_Rnm(core.Question):
    name = 'ある内積とそのノルム（行列のベクトル空間）'
    def __init__(self, emin=-3, emax=3, dmin=4, dmax=4):
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 生成するベクトルの次元の範囲
        self.dim_min = dmin
        self.dim_max = dmax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='行列のある内積とそのノルム', quiz_number=_quiz_number)
        ins = linear_algebra.InnerSpace(domain='Rnm')
        _dimension = random.randint(self.dim_min, self.dim_max)
        _vectA = [random.randint(self.elem_min, self.elem_max) for i in range(_dimension)]
        _vectB = [random.randint(self.elem_min, self.elem_max) for i in range(_dimension)]
        quiz.quiz_identifier = hash(str(_vectA) + str(_vectB))
        # 正答の選択肢の生成
        _inner_product = ins.inner_product(_vectA, _vectB)
        _normA = ins.norm(_vectA)
        _normB = ins.norm(_vectB)
        quiz.data = [_dimension, _vectA, _vectB, _inner_product, _normA, _normB]
        ans = { 'fraction': 100, 'data': [_inner_product, _normA, _normB] }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_dimension, _vectA, _vectB, _inner_product, _normA, _normB] = quiz.data
        ins = linear_algebra.InnerSpace(domain='Rnm')
        ans['feedback'] = '今回の内積は，きちんと内積の条件を満たしています。'
        ans['data'] = [r'この内積はこのベクトル空間では内積の条件を満たさず，定義されない。']
        answers.append(dict(ans))
        incorrect_inner_product = -_inner_product
        incorrect_normA = _normA
        incorrect_normB = _normB
        if incorrect_inner_product != _inner_product:
            ans['feedback'] = '与えられた内積の定義に基づいて計算し直しましょう。'
            ans['data'] = [incorrect_inner_product, incorrect_normA, incorrect_normB]
            answers.append(dict(ans))
        incorrect_inner_product = _inner_product
        incorrect_normA = ins.inner_product(_vectA, _vectA)
        incorrect_normB = ins.inner_product(_vectB, _vectB)
        if incorrect_inner_product != _inner_product or incorrect_normA != _normA or incorrect_normB != _normB:
            ans['feedback'] = 'ノルムは，自分自身との内積の二乗根です。計算し直しましょう。'
            ans['data'] = [incorrect_inner_product, incorrect_normA, incorrect_normB]
            answers.append(dict(ans))
        incorrect_inner_product = -_inner_product
        if incorrect_inner_product != _inner_product or incorrect_normA != _normA or incorrect_normB != _normB:
            ans['feedback'] = 'ノルムは，自分自身との内積の二乗根です。計算し直しましょう。'
            ans['data'] = [incorrect_inner_product, incorrect_normA, incorrect_normB]
            answers.append(dict(ans))
        incorrect_inner_product = sum(_vectA) + sum(_vectB)
        incorrect_normA = abs(sum(_vectA))
        incorrect_normB = abs(sum(_vectB))
        if incorrect_inner_product != _inner_product or incorrect_normA != _normA or incorrect_normB != _normB:
            ans['feedback'] = '与えられた内積の定義に基づいて計算し直しましょう。'
            ans['data'] = [incorrect_inner_product, incorrect_normA, incorrect_normB]
            answers.append(dict(ans))
        incorrect_inner_product = -_inner_product
        if incorrect_inner_product != _inner_product or incorrect_normA != _normA or incorrect_normB != _normB:
            ans['feedback'] = '与えられた内積の定義に基づいて計算し直しましょう。'
            ans['data'] = [incorrect_inner_product, incorrect_normA, incorrect_normB]
            answers.append(dict(ans))
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_dimension, _vectA, _vectB, _inner_product, _normA, _normB] = quiz.data
        ins = linear_algebra.InnerSpace(domain='Rnm')
        _text = r'ベクトル空間' + ins.str_domain(is_latex_closure=True) + r'の内積を'
        _text += ins.str_inner_product(is_latex_closure=True)
        _text += r'とする。このとき，次の2つのベクトルの内積とそれぞれのノルムとして適切なものを選択してください。<br />'
        _text += ins.str_vectors([_vectA, _vectB])
        return _text
    def answer_text(self, ans):
        if len(ans['data']) > 1:
            [_inner_product, _normA, _normB] = ans['data']
            _text = r'内積: \(' + sympy.latex(_inner_product) + r'\), '
            _text += r'ノルム: \(' + sympy.latex(_normA) + r',\;' + sympy.latex(_normB) + r'\)'
        else:
            _text = ans['data'][0]
        return _text


# In[17]:


if __name__ == "__main__":
    q = some_inner_product_with_norm_in_Rnm()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[18]:


if __name__ == "__main__":
    pass
    #qz.save('some_inner_product_with_norm_in_Rnm.xml')


# ## expanding inner expression with numerical values

# In[3]:


class expanding_inner_expression_with_numerical_values(core.Question):
    name = '内積やノルムを展開することで値を求める'
    def __init__(self, emin=-2, emax=2, dmin=2, dmax=3):
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 生成するベクトルの個数の範囲
        self.dim_min = dmin
        self.dim_max = dmax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='内積やノルムの展開計算', quiz_number=_quiz_number)
        ins = linear_algebra.InnerSpace(domain='Rn')
        _num_of_vectors = random.randint(self.dim_min, self.dim_max)
        _vecs = [[linear_algebra.nonzero_randint(self.elem_min, self.elem_max) for i in range(2)] for j in range(_num_of_vectors)]
        _inners = [[0 for i in range(_num_of_vectors)] for j in range(_num_of_vectors)]
        for i in range(_num_of_vectors):
            for j in range(_num_of_vectors):
                _inners[i][j] = ins.inner_product(_vecs[i], _vecs[j])
        _expr = ins.generate_symbolic(_num_of_vectors, self.elem_min, self.elem_max)
        quiz.quiz_identifier = hash(str(_expr))
        # 正答の選択肢の生成
        _value = ins.eval_symbolic(_expr, _inners)
        quiz.data = [_num_of_vectors, _inners, _expr, _value]
        ans = { 'fraction': 100, 'data': _value }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_num_of_vectors, _inners, _expr, _value] = quiz.data
        ins = linear_algebra.InnerSpace(domain=None)
        ans['feedback'] = 'どのような内積であっても内積の性質に基づき計算が出来ます。'
        ans['data'] = r'具体的に内積やノルムが定義されていないので計算不能。'
        answers.append(dict(ans))
        loop = 0
        while len(answers) < size and loop < 10:
            incorrect_value = ins.eval_symbolic(_expr, _inners, wrong=True)
            if _value != incorrect_value:
                ans['feedback'] = '双線形性に基づいて展開して計算を行いますが，スカラー倍や和の残りなど取りこぼしがあると計算を間違います。丁寧に再計算しましょう。'
                ans['data'] = incorrect_value
                answers.append(dict(ans))
            answers = common.answer_union(answers)
            loop += 1
        while len(answers) < size:
            incorrect_value = _value + linear_algebra.nonzero_randint(-5,5)
            if _value != incorrect_value:
                ans['feedback'] = '丁寧に再計算しましょう。'
                ans['data'] = incorrect_value
                answers.append(dict(ans))
            answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_num_of_vectors, _inners, _expr, _value] = quiz.data
        ins = linear_algebra.InnerSpace(domain=None)
        _text = r'\(\mathbb{R}\)上の内積空間\(V\)のいくつかのベクトルの内積を計算したところ，下記の値を得たとする。<br />'
        for i in range(_num_of_vectors):
            for j in range(i, _num_of_vectors):
                _text += r'\(\left(\vec{v_{' + str(i+1) + r'}},\vec{v_{' + str(j+1) + r'}}\right)=' + sympy.latex(_inners[i][j]) + r'\)'
                if i < _num_of_vectors - 1 or j < _num_of_vectors - 1:
                    _text += r', '
        _text += r'<br />このとき，次の値として適切なものを選択してください。<br />'
        _text += ins.str_symbolic(_expr, is_latex_closure=True, variable='v')
        return _text
    def answer_text(self, ans):
        if isinstance(ans['data'], str):
            _text = ans['data']
        else:
            _text = r'\(' + sympy.latex(ans['data']) + r'\)'
        return _text


# In[4]:


if __name__ == "__main__":
    q = expanding_inner_expression_with_numerical_values()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[5]:


if __name__ == "__main__":
    pass
    #qz.save('expanding_inner_expression_with_numerical_values.xml')


# ## expanding inner expression without numerical values

# In[10]:


class expanding_inner_expression_without_numerical_values(core.Question):
    name = '内積やノルムを展開することで展開後の式を求める'
    def __init__(self, emin=-2, emax=2, dmin=2, dmax=3, expmin=1, expmax=3):
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 生成するベクトルの個数の範囲
        self.dim_min = dmin
        self.dim_max = dmax
        # 展開するネストの回数
        self.exp_min = expmin
        self.exp_max = expmax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='内積やノルムの展開式の計算', quiz_number=_quiz_number)
        ins = linear_algebra.InnerSpace(domain='Rn')
        _num_of_vectors = random.randint(self.dim_min, self.dim_max)
        _expr = ins.generate_symbolic(_num_of_vectors, self.elem_min, self.elem_max, norm_ratio=0)
        quiz.quiz_identifier = hash(str(_expr))
        # 正答の選択肢の生成
        _expanded_expr = ins.expand_symbolic(_expr, nest=random.randint(self.exp_min, self.exp_max))
        quiz.data = [_num_of_vectors, _expr, _expanded_expr]
        ans = { 'fraction': 100, 'data': _expanded_expr }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_num_of_vectors, _expr, _expanded_expr] = quiz.data
        ins = linear_algebra.InnerSpace(domain='Rn')
        _vecs = [[linear_algebra.nonzero_randint(-100,100) for i in range(5)] for j in range(_num_of_vectors)]
        _inners = [[0 for i in range(_num_of_vectors)] for j in range(_num_of_vectors)]
        for i in range(_num_of_vectors):
            for j in range(i,_num_of_vectors):
                _inners[i][j] = ins.inner_product(_vecs[i], _vecs[j])
        _value = ins.eval_symbolic(_expanded_expr, _inners)
        loop = 0
        while len(answers) < size and loop < 32:
            incorrect_expanded_expr = ins.expand_symbolic(_expr, wrong=True, nest=random.randint(self.exp_min, self.exp_max))
            incorrect_value = ins.eval_symbolic(incorrect_expanded_expr, _inners)
            if _expanded_expr != incorrect_expanded_expr and _value != incorrect_value:
                ans['feedback'] = '双線形性に基づいて展開して計算を行いますが，スカラー倍や和の残りなど取りこぼしがあると計算を間違います。丁寧に再計算しましょう。'
                ans['data'] = incorrect_expanded_expr
                answers.append(dict(ans))
            answers = common.answer_union(answers)
            loop += 1
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_num_of_vectors, _expr, _expanded_expr] = quiz.data
        ins = linear_algebra.InnerSpace(domain=None)
        _text = r'\(\mathbb{R}\)上の内積空間\(V\)の次の内積に等しい式として適切なものを選択してください。<br />'
        _text += ins.str_symbolic(_expr, is_latex_closure=True, variable='v')
        return _text
    def answer_text(self, ans):
        ins = linear_algebra.InnerSpace(domain=None)
        _text = ins.str_symbolic(ans['data'], is_latex_closure=True, variable='v', parenthesis=False)
        return _text


# In[11]:


if __name__ == "__main__":
    q = expanding_inner_expression_without_numerical_values()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[12]:


if __name__ == "__main__":
    pass
    #qz.save('expanding_inner_expression_without_numerical_values.xml')


# ## basis of orthogonal complement

# In[23]:


class basis_of_orthogonal_complement(core.Question):
    name = '直交補空間の一組の基底を選択する'
    def __init__(self, emin=-3, emax=3, dmin=2, dmax=4, rdefimin=1, rdefimax=2, rdunmin=1, rdunmax=2):
        # 生成する個々の数の範囲
        self.emin = emin
        self.emax = emax
        # 生成するベクトルの次元の範囲
        self.dmin = dmin
        self.dmax = dmax
        # 生成するベクトルの次元からのランク落ちの範囲
        self.rdefimin = rdefimin
        self.rdefimax = rdefimax
        # 生成するベクトルの基底からの余計なベクトルの範囲
        self.rdunmin = rdunmin
        self.rdunmax = rdunmax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='直交補空間の基底', quiz_number=_quiz_number)
        oc = linear_algebra.OrthogonalComplement()
        oc.set_dimension_range(self.dmin, self.dmax)
        oc.set_element_range(self.emin, self.emax)
        oc.set_rank_deficient_range(self.rdefimin, self.rdefimax)
        oc.set_redundant_range(self.rdunmin, self.rdunmax)
        oc.generate()
        quiz.data = [oc]
        quiz.quiz_identifier = hash(oc)
        # 正答の選択肢の生成
        ans = { 'fraction': 100, 'data': oc.basis }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [oc] = quiz.data
        if len(oc.basis) > 1:
            incorrect_answer = random.sample(oc.basis, len(oc.basis)-1)
            ans['feedback'] = 'ケアレスミスの可能性があります。再度掃き出し結果を確認しましょう。'
            ans['data'] = incorrect_answer
            answers.append(dict(ans))
        if oc.is_for_solution_space:
            _null_space = sympy.Matrix(oc.given_space).nullspace(simplify=True)
            incorrect_answer = [sympy.matrix2numpy(m.transpose()).tolist()[0] for m in _null_space]
            if not oc.is_a_basis(incorrect_answer):
                ans['feedback'] = '与えられた部分空間の基底ではなく，直交補空間の基底を求めてください。'
                ans['data'] = incorrect_answer
                answers.append(dict(ans))
        else:
            _row_space = sympy.Matrix(oc.given_space).rowspace(simplify=True)
            incorrect_answer = [sympy.matrix2numpy(m).tolist()[0] for m in _row_space]
            if not oc.is_a_basis(incorrect_answer):
                ans['feedback'] = '与えられた部分空間の基底ではなく，直交補空間の基底を求めてください。'
                ans['data'] = incorrect_answer
                answers.append(dict(ans))
        if len(incorrect_answer) > 1:
            incorrect_answer = random.sample(incorrect_answer, len(incorrect_answer)-1)
            ans['feedback'] = '与えられた部分空間の基底もどきではなく，直交補空間の基底を求めてください。'
            ans['data'] = incorrect_answer
            answers.append(dict(ans))
        incorrect_answer = oc.given_space
        if not oc.is_a_basis(incorrect_answer):
            ans['feedback'] = '与えられた部分空間のベクトルをそのまま書かずに，直交補空間の基底を求めてください。'
            ans['data'] = incorrect_answer
            answers.append(dict(ans))
        if len(incorrect_answer) > 1:
            incorrect_answer = random.sample(incorrect_answer, len(incorrect_answer)-1)
            ans['feedback'] = '与えられた部分空間のベクトルをそのまま書かずに，直交補空間の基底を求めてください。'
            ans['data'] = incorrect_answer
            answers.append(dict(ans))
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)       
        return answers    
    def question_text(self, quiz):
        _text = '次の部分空間の直交補空間の基底として適切なものを選択してください。なお，内積は標準内積を用いてください。<br />'
        _text += quiz.data[0].str_given_space(is_latex_closure=True)
        return  _text
    def answer_text(self, ans):
        return self._str_list_vectors(ans['data'])
    def _str_list_vectors(self, vecs):
        _text = r'\( \left\{'
        for i in range(len(vecs)):
            _text += sympy.latex(sympy.Matrix([vecs[i]]).transpose(), mat_delim='', mat_str='pmatrix')
            if i < len(vecs) - 1:
                _text += r',\;'        
        return _text + r'\right\} \)'


# In[24]:


if __name__ == "__main__":
    q = basis_of_orthogonal_complement()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[25]:


if __name__ == "__main__":
    pass
    #qz.save('basis_of_orthogonal_complement.xml')


# ## orthonomal basis in Rn with standard inner product

# In[47]:


class orthonomal_basis_in_Rn_with_standard_inner_product(core.Question):
    name = '正規直交基底の選択（数ベクトル空間，標準内積）'
    def __init__(self, emin=-2, emax=2, dmin=2, dmax=3, zratio=0.5):
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 生成するベクトルの次元の範囲
        self.dim_min = dmin
        self.dim_max = dmax
        # 要素を零とする割合
        self.zero_ratio = zratio
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='正規直交基底の選択', quiz_number=_quiz_number)
        ins = linear_algebra.InnerSpace(domain='Rn')
        _dimension = random.randint(self.dim_min, self.dim_max)
        _orthonomal_basis = [[1]]
        _mat = sympy.Matrix([[0]])
        while _mat.rank() < _dimension or max([abs(e) for e in linear_algebra.flatten_list(_orthonomal_basis)]) == 1:
            _vecs = [[random.randint(self.elem_min, self.elem_max) if random.random() > self.zero_ratio else 0 for i in range(_dimension)] for j in range(_dimension)]
            _orthonomal_basis = ins.orthonomalize(_vecs)
            _mat = sympy.Matrix(_vecs)
        # 正答の選択肢の生成
        quiz.quiz_identifier = hash(str(_orthonomal_basis))
        quiz.data = [_dimension, _vecs, _orthonomal_basis]
        ans = { 'fraction': 100, 'data': _orthonomal_basis }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_dimension, _vecs, _orthonomal_basis] = quiz.data
        ins = linear_algebra.InnerSpace(domain='Rn')
        ans['feedback'] = '含まれているので，直交性と正規性をそれぞれ確認してください。'
        ans['data'] = r'正規直交基底は含まれていない。'
        answers.append(dict(ans))
        _elems = list(range(0, max(abs(self.elem_min), abs(self.elem_max)) + 1))
        _matE = sympy.eye(_dimension)
        incorrect_basis = ins.orthogonalize(_vecs)
        _mat = sympy.Matrix(incorrect_basis)
        if _mat*_mat.transpose() != _matE or _mat.transpose()*_mat != _matE:
            ans['feedback'] = '正規直交基底とは，直交基底であって，かつ，正規である必要があります。'
            ans['data'] = incorrect_basis
            answers.append(dict(ans))
        if _dimension > 1:
            incorrect_basis = _vecs[1:]
            _mu = [random.choice([-2,-1,1,2]) for i in range(len(incorrect_basis))]
            incorrect_basis.append([sum([_mu[j]*incorrect_basis[j][i] for j in range(len(incorrect_basis))]) for i in range(_dimension)])
            _mat = sympy.Matrix(incorrect_basis)
            if _mat.rank() < _dimension:
                ans['feedback'] = '正規直交基底とは，まずは基底である必要があります。'
                ans['data'] = incorrect_basis
                answers.append(dict(ans))
            incorrect_basis = ins.orthonomalize(_vecs)[1:]
            _mu = random.choice(_elems + [-e for e in _elems] + [sympy.sqrt(e) for e in _elems] + [-sympy.sqrt(e) for e in _elems])
            _vec = random.choice(incorrect_basis)
            incorrect_basis.append([_mu*e for e in _vec])
            _mat = sympy.Matrix(incorrect_basis)
            if _mat.rank() < _dimension:
                ans['feedback'] = '正規直交基底とは，まずは基底である必要があります。'
                ans['data'] = incorrect_basis
                answers.append(dict(ans))
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_dimension, _vecs, _orthonomal_basis] = quiz.data
        _text = r'数ベクトル空間\(\mathbb{R}^{' + str(_dimension) + r'}\)の正規直交基底として適切なものを選択してください。なお，内積は標準内積を用いてください。'
        return _text
    def answer_text(self, ans):
        if isinstance(ans['data'], str):
            _text = ans['data']
        else:
            ins = linear_algebra.InnerSpace(domain='Rn')
            _text = r'\(\left\{' + ins.str_vectors(ans['data'], is_latex_closure=False, variable=None)  + r'\right\}\)'
        return _text


# In[49]:


if __name__ == "__main__":
    q = orthonomal_basis_in_Rn_with_standard_inner_product()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=2)


# In[50]:


if __name__ == "__main__":
    pass
    #qz.save('orthonomal_basis_in_Rn_with_standard_inner_product.xml')


# ## orthogonalization of Gram-Schmidt process in Rn with standard inner product

# In[77]:


class orthogonalization_process_in_Rn_with_standard_inner_product(core.Question):
    name = 'グラムシュミットの直交化部分の計算（数ベクトル空間，標準内積）'
    def __init__(self, emin=-2, emax=2, dmin=2, dmax=3, zratio=0.5):
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 生成するベクトルの次元の範囲
        self.dim_min = dmin
        self.dim_max = dmax
        # 要素を零とする割合
        self.zero_ratio = zratio
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='直交化部分の計算（Rn，標準内積）', quiz_number=_quiz_number)
        ins = linear_algebra.InnerSpace(domain='Rn')
        _dimension = random.randint(self.dim_min, self.dim_max)
        _orthonomal_basis = [[1]]
        _mat = sympy.Matrix([[0]])
        while _mat.rank() < _dimension or max([abs(e) for e in linear_algebra.flatten_list(_orthonomal_basis)]) == 1:
            _vecs = [[random.randint(self.elem_min, self.elem_max) if random.random() > self.zero_ratio else 0 for i in range(_dimension)] for j in range(_dimension)]
            _orthonomal_basis = ins.orthonomalize(_vecs)
            _mat = sympy.Matrix(_vecs)
        # 正答の選択肢の生成
        _orthogonal_basis = ins.orthogonalize(_vecs)
        quiz.quiz_identifier = hash(str(_orthogonal_basis))
        quiz.data = [_dimension, _vecs, _orthogonal_basis, _orthonomal_basis ]
        ans = { 'fraction': 100, 'data': _orthogonal_basis[-1] }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_dimension, _vecs, _orthogonal_basis, _orthonomal_basis ] = quiz.data
        ins = linear_algebra.InnerSpace(domain='Rn')
        incorrect_vec = _vecs[-1]
        if incorrect_vec != _orthogonal_basis[-1]:
            ans['feedback'] = '直交化の手順を確認しましょう。正射影分を取り除く必要があります。'
            ans['data'] = incorrect_vec
            answers.append(dict(ans))
        if _dimension > 2:
            incorrect_vec = _vecs[-1]
            for v in _orthogonal_basis[:-2]:
                _mu = ins.inner_product(v, _vecs[-1]) * sympy.Pow(ins.inner_product(v, v), -1)
                incorrect_vec = [incorrect_vec[j] - _mu*v[j] for j in range(len(v))]
            if incorrect_vec != _orthogonal_basis[-1]:
                ans['feedback'] = '直交化の手順を確認しましょう。既に直交化済みのベクトル毎に正射影分を取り除く必要があります。'
                ans['data'] = incorrect_vec
                answers.append(dict(ans))
        incorrect_vec = _vecs[-1]
        for v in _orthogonal_basis[:-1]:
            _mu = ins.inner_product(v, _vecs[-1])# * sympy.Pow(ins.inner_product(v, v), -1)
            incorrect_vec = [incorrect_vec[j] - _mu*v[j] for j in range(len(v))]
        if incorrect_vec != _orthogonal_basis[-1]:
            ans['feedback'] = '直交化の手順を確認しましょう。正規化前なので正射影の大きさは内積2つの有理式になっているはずです。'
            ans['data'] = incorrect_vec
            answers.append(dict(ans))
        incorrect_vec = _vecs[-1]
        for v in _orthogonal_basis[:-1]:
            if ins.inner_product(v,  _vecs[-1]) == 0:
                continue
            _mu = ins.inner_product(v, v) * sympy.Pow(ins.inner_product(v,  _vecs[-1]), -1)
            incorrect_vec = [incorrect_vec[j] - _mu*v[j] for j in range(len(v))]
        if incorrect_vec != _orthogonal_basis[-1]:
            ans['feedback'] = '直交化の手順を確認しましょう。正射影の大きさの2つの内積の役割が逆転しています。'
            ans['data'] = incorrect_vec
            answers.append(dict(ans))
        incorrect_vec = _orthonomal_basis[-1]
        if incorrect_vec != _orthogonal_basis[-1]:
            ans['feedback'] = '直交化のみを行った結果を選んでください。'
            ans['data'] = incorrect_vec
            answers.append(dict(ans))
        answers = common.answer_union(answers)
        while len(answers) < size:
            incorrect_vec = [_orthogonal_basis[-1][i] + random.choice([-1,0,1])  for i in range(_dimension)]
            if incorrect_vec != _orthogonal_basis[-1]:
                ans['feedback'] = '計算を丁寧に行いましょう。'
                ans['data'] = incorrect_vec
                answers.append(dict(ans))
            answers = common.answer_union(answers)            
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_dimension, _vecs, _orthogonal_basis, _orthonomal_basis ] = quiz.data
        ins = linear_algebra.InnerSpace(domain='Rn')
        _text = r'ベクトル空間\(\mathbb{R}^{' + str(_dimension) + r'}\)の内積を標準内積とします。'
        _text += r'このとき，次のベクトル'
        for i in range(_dimension):
            _text += r'\(\vec{w_{' + str(i+1) + r'}}\)'
            if i < _dimension - 1:
                _text += r','
        _text += r'を，グラムシュミットの方法で正規直交化することを考えます。'
        _text += r'ひとまず，直交化部分のみ（正規化を含まない）を途中まで行ったところ，ベクトル'
        for i in range(_dimension - 1):
            _text += r'\(\vec{v_{' + str(i+1) + r'}}\)'
            if i < _dimension - 2:
                _text += r','
        _text += r'が得られました。次の直交ベクトルを求める手順を行った場合に得られるベクトルとしてもっとも適切なものを選択してください。<br />'
        _text += ins.str_vectors(_vecs, is_latex_closure=True, variable=r'w') + r'<br />'
        _text += ins.str_vectors(_orthogonal_basis[:-1], is_latex_closure=True, variable=r'v')
        return _text
    def answer_text(self, ans):
        if isinstance(ans['data'], str):
            _text = ans['data']
        else:
            ins = linear_algebra.InnerSpace(domain='Rn')
            _text = ins.str_vectors([ans['data']], is_latex_closure=True, variable=None)
        return _text


# In[78]:


if __name__ == "__main__":
    q = orthogonalization_process_in_Rn_with_standard_inner_product()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=2)


# In[79]:


if __name__ == "__main__":
    pass
    #qz.save('orthogonalization_process_in_Rn_with_standard_inner_product.xml')


# ## normalization of Gram-Schmidt process in Rn with standard inner product

# In[90]:


class normalization_process_in_Rn_with_standard_inner_product(core.Question):
    name = 'グラムシュミットの正規化部分の計算（数ベクトル空間，標準内積）'
    def __init__(self, emin=-2, emax=2, dmin=2, dmax=3, zratio=0.5):
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 生成するベクトルの次元の範囲
        self.dim_min = dmin
        self.dim_max = dmax
        # 要素を零とする割合
        self.zero_ratio = zratio
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='正規化部分の計算（Rn，標準内積）', quiz_number=_quiz_number)
        ins = linear_algebra.InnerSpace(domain='Rn')
        _dimension = random.randint(self.dim_min, self.dim_max)
        _orthonomal_basis = [[1]]
        _mat = sympy.Matrix([[0]])
        while _mat.rank() < _dimension or max([abs(e) for e in linear_algebra.flatten_list(_orthonomal_basis)]) == 1:
            _vecs = [[random.randint(self.elem_min, self.elem_max) if random.random() > self.zero_ratio else 0 for i in range(_dimension)] for j in range(_dimension)]
            _orthonomal_basis = ins.orthonomalize(_vecs)
            _mat = sympy.Matrix(_vecs)
        # 正答の選択肢の生成
        _orthogonal_basis = ins.orthogonalize(_vecs)
        quiz.quiz_identifier = hash(str(_orthogonal_basis))
        quiz.data = [_dimension, _vecs, _orthogonal_basis, _orthonomal_basis ]
        ans = { 'fraction': 100, 'data': _orthonomal_basis }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_dimension, _vecs, _orthogonal_basis, _orthonomal_basis ] = quiz.data
        ins = linear_algebra.InnerSpace(domain='Rn')
        incorrect_basis = _vecs
        if sympy.Matrix(incorrect_basis) != sympy.Matrix(_orthonomal_basis):
            ans['feedback'] = '直交化の次に行う正規化は，ノルムを1にする操作です。最初のベクトルに戻さないでください。'
            ans['data'] = incorrect_basis
            answers.append(dict(ans))
        incorrect_basis = _orthogonal_basis
        if sympy.Matrix(incorrect_basis) != sympy.Matrix(_orthonomal_basis):
            ans['feedback'] = '直交化の次に行う正規化は，ノルムを1にする操作です。そのままではいけません。'
            ans['data'] = incorrect_basis
            answers.append(dict(ans))
        _truefalse = [True for i in range(_dimension - 1)] + [False]
        _truefalse = random.sample(_truefalse, _dimension)        
        incorrect_basis = [_orthonomal_basis[i] if _truefalse[i] else _orthogonal_basis[i] for i in range(_dimension)]
        if sympy.Matrix(incorrect_basis) != sympy.Matrix(_orthonomal_basis):
            ans['feedback'] = '全てのベクトルを正規化してください。'
            ans['data'] = incorrect_basis
            answers.append(dict(ans)) 
        _truefalse = random.sample(_truefalse, _dimension)                
        incorrect_basis = [_orthonomal_basis[i] if _truefalse[i] else _orthogonal_basis[i] for i in range(_dimension)]
        if sympy.Matrix(incorrect_basis) != sympy.Matrix(_orthonomal_basis):
            ans['feedback'] = '全てのベクトルを正規化してください。'
            ans['data'] = incorrect_basis
            answers.append(dict(ans))        
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_dimension, _vecs, _orthogonal_basis, _orthonomal_basis ] = quiz.data
        ins = linear_algebra.InnerSpace(domain='Rn')
        _text = r'ベクトル空間\(\mathbb{R}^{' + str(_dimension) + r'}\)の内積を標準内積とします。'
        _text += r'このとき，次のベクトル'
        for i in range(_dimension):
            _text += r'\(\vec{w_{' + str(i+1) + r'}}\)'
            if i < _dimension - 1:
                _text += r','
        _text += r'を，グラムシュミットの方法で正規直交化することを考えます。'
        _text += r'ひとまず，直交化部分のみ（正規化を含まない）を行ったところ，ベクトル'
        for i in range(_dimension):
            _text += r'\(\vec{v_{' + str(i+1) + r'}}\)'
            if i < _dimension - 1:
                _text += r','
        _text += r'が得られました。次に正規化（正規直交化する残りの手順）を行った場合に得られるベクトルとしてもっとも適切なものを選択してください。<br />'
        _text += ins.str_vectors(_vecs, is_latex_closure=True, variable=r'w') + r'<br />'
        _text += ins.str_vectors(_orthogonal_basis, is_latex_closure=True, variable=r'v')
        return _text
    def answer_text(self, ans):
        if isinstance(ans['data'], str):
            _text = ans['data']
        else:
            ins = linear_algebra.InnerSpace(domain='Rn')
            _text = ins.str_vectors(ans['data'], is_latex_closure=True, variable=None)
        return _text


# In[93]:


if __name__ == "__main__":
    q = normalization_process_in_Rn_with_standard_inner_product()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=2)


# In[94]:


if __name__ == "__main__":
    pass
    #qz.save('normalization_process_in_Rn_with_standard_inner_product.xml')


# ## orthogonalization of Gram-Schmidt process in Rn with some inner product

# In[100]:


class orthogonalization_process_in_Rn_with_some_inner_product(core.Question):
    name = 'グラムシュミットの直交化部分の計算（数ベクトル空間，ある内積（標準内積でない））'
    def __init__(self, emin=-2, emax=2, dmin=2, dmax=3, zratio=0.5):
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 生成するベクトルの次元の範囲
        self.dim_min = dmin
        self.dim_max = dmax
        # 要素を零とする割合
        self.zero_ratio = zratio
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='直交化部分の計算（Rn，ある内積）', quiz_number=_quiz_number)
        ins = linear_algebra.InnerSpace(domain='aRn')
        _dimension = random.randint(self.dim_min, self.dim_max)
        _orthonomal_basis = [[1]]
        _mat = sympy.Matrix([[0]])
        while _mat.rank() < _dimension or max([abs(e) for e in linear_algebra.flatten_list(_orthonomal_basis)]) == 1:
            _vecs = [[random.randint(self.elem_min, self.elem_max) if random.random() > self.zero_ratio else 0 for i in range(_dimension)] for j in range(_dimension)]
            _orthonomal_basis = ins.orthonomalize(_vecs)
            _mat = sympy.Matrix(_vecs)
        # 正答の選択肢の生成
        _orthogonal_basis = ins.orthogonalize(_vecs)
        quiz.quiz_identifier = hash(str(_orthogonal_basis))
        quiz.data = [_dimension, _vecs, _orthogonal_basis, _orthonomal_basis ]
        ans = { 'fraction': 100, 'data': _orthogonal_basis[-1] }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_dimension, _vecs, _orthogonal_basis, _orthonomal_basis ] = quiz.data
        ins = linear_algebra.InnerSpace(domain='aRn')
        incorrect_vec = _vecs[-1]
        if incorrect_vec != _orthogonal_basis[-1]:
            ans['feedback'] = '直交化の手順を確認しましょう。正射影分を取り除く必要があります。'
            ans['data'] = incorrect_vec
            answers.append(dict(ans))
        another_ins = linear_algebra.InnerSpace(domain='Rn')
        incorrect_vec = _vecs[-1]
        for v in _orthogonal_basis[:-1]:
            _mu = another_ins.inner_product(v, _vecs[-1]) * sympy.Pow(another_ins.inner_product(v, v), -1)
            incorrect_vec = [incorrect_vec[j] - _mu*v[j] for j in range(len(v))]
        if incorrect_vec != _orthogonal_basis[-1]:
            ans['feedback'] = '内積の定義に注意してください。標準内積ではありません。'
            ans['data'] = incorrect_vec
            answers.append(dict(ans))
        if _dimension > 2:
            incorrect_vec = _vecs[-1]
            for v in _orthogonal_basis[:-2]:
                _mu = ins.inner_product(v, _vecs[-1]) * sympy.Pow(ins.inner_product(v, v), -1)
                incorrect_vec = [incorrect_vec[j] - _mu*v[j] for j in range(len(v))]
            if incorrect_vec != _orthogonal_basis[-1]:
                ans['feedback'] = '直交化の手順を確認しましょう。既に直交化済みのベクトル毎に正射影分を取り除く必要があります。'
                ans['data'] = incorrect_vec
                answers.append(dict(ans))
        incorrect_vec = _vecs[-1]
        for v in _orthogonal_basis[:-1]:
            _mu = ins.inner_product(v, _vecs[-1])# * sympy.Pow(ins.inner_product(v, v), -1)
            incorrect_vec = [incorrect_vec[j] - _mu*v[j] for j in range(len(v))]
        if incorrect_vec != _orthogonal_basis[-1]:
            ans['feedback'] = '直交化の手順を確認しましょう。正規化前なので正射影の大きさは内積2つの有理式になっているはずです。'
            ans['data'] = incorrect_vec
            answers.append(dict(ans))
        incorrect_vec = _vecs[-1]
        for v in _orthogonal_basis[:-1]:
            if ins.inner_product(v,  _vecs[-1]) == 0:
                continue
            _mu = ins.inner_product(v, v) * sympy.Pow(ins.inner_product(v,  _vecs[-1]), -1)
            incorrect_vec = [incorrect_vec[j] - _mu*v[j] for j in range(len(v))]
        if incorrect_vec != _orthogonal_basis[-1]:
            ans['feedback'] = '直交化の手順を確認しましょう。正射影の大きさの2つの内積の役割が逆転しています。'
            ans['data'] = incorrect_vec
            answers.append(dict(ans))
        incorrect_vec = _orthonomal_basis[-1]
        if incorrect_vec != _orthogonal_basis[-1]:
            ans['feedback'] = '直交化のみを行った結果を選んでください。'
            ans['data'] = incorrect_vec
            answers.append(dict(ans))
        answers = common.answer_union(answers)
        while len(answers) < size:
            incorrect_vec = [_orthogonal_basis[-1][i] + random.choice([-1,0,1])  for i in range(_dimension)]
            if incorrect_vec != _orthogonal_basis[-1]:
                ans['feedback'] = '計算を丁寧に行いましょう。'
                ans['data'] = incorrect_vec
                answers.append(dict(ans))
            answers = common.answer_union(answers)            
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_dimension, _vecs, _orthogonal_basis, _orthonomal_basis ] = quiz.data
        ins = linear_algebra.InnerSpace(domain='aRn')
        _text = r'ベクトル空間\(\mathbb{R}^{' + str(_dimension) + r'}\)の内積を'
        _text += ins.str_inner_product(is_latex_closure=True)
        _text += r'とします。このとき，次のベクトル'
        for i in range(_dimension):
            _text += r'\(\vec{w_{' + str(i+1) + r'}}\)'
            if i < _dimension - 1:
                _text += r','
        _text += r'を，グラムシュミットの方法で正規直交化することを考えます。'
        _text += r'ひとまず，直交化部分のみ（正規化を含まない）を途中まで行ったところ，ベクトル'
        for i in range(_dimension - 1):
            _text += r'\(\vec{v_{' + str(i+1) + r'}}\)'
            if i < _dimension - 2:
                _text += r','
        _text += r'が得られました。次の直交ベクトルを求める手順を行った場合に得られるベクトルとしてもっとも適切なものを選択してください。<br />'
        _text += ins.str_vectors(_vecs, is_latex_closure=True, variable=r'w') + r'<br />'
        _text += ins.str_vectors(_orthogonal_basis[:-1], is_latex_closure=True, variable=r'v')
        return _text
    def answer_text(self, ans):
        if isinstance(ans['data'], str):
            _text = ans['data']
        else:
            ins = linear_algebra.InnerSpace(domain='aRn')
            _text = ins.str_vectors([ans['data']], is_latex_closure=True, variable=None)
        return _text


# In[103]:


if __name__ == "__main__":
    q = orthogonalization_process_in_Rn_with_some_inner_product()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=2)


# In[104]:


if __name__ == "__main__":
    pass
    #qz.save('orthogonalization_process_in_Rn_with_some_inner_product.xml')


# ## normalization of Gram-Schmidt process in Rn with some inner product

# In[109]:


class normalization_process_in_Rn_with_some_inner_product(core.Question):
    name = 'グラムシュミットの正規化部分の計算（数ベクトル空間，ある内積（標準内積でない））'
    def __init__(self, emin=-2, emax=2, dmin=2, dmax=3, zratio=0.5):
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 生成するベクトルの次元の範囲
        self.dim_min = dmin
        self.dim_max = dmax
        # 要素を零とする割合
        self.zero_ratio = zratio
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='正規化部分の計算（Rn，ある内積）', quiz_number=_quiz_number)
        ins = linear_algebra.InnerSpace(domain='aRn')
        _dimension = random.randint(self.dim_min, self.dim_max)
        _orthonomal_basis = [[1]]
        _mat = sympy.Matrix([[0]])
        while _mat.rank() < _dimension or max([abs(e) for e in linear_algebra.flatten_list(_orthonomal_basis)]) == 1:
            _vecs = [[random.randint(self.elem_min, self.elem_max) if random.random() > self.zero_ratio else 0 for i in range(_dimension)] for j in range(_dimension)]
            _orthonomal_basis = ins.orthonomalize(_vecs)
            _mat = sympy.Matrix(_vecs)
        # 正答の選択肢の生成
        _orthogonal_basis = ins.orthogonalize(_vecs)
        quiz.quiz_identifier = hash(str(_orthogonal_basis))
        quiz.data = [_dimension, _vecs, _orthogonal_basis, _orthonomal_basis ]
        ans = { 'fraction': 100, 'data': _orthonomal_basis }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_dimension, _vecs, _orthogonal_basis, _orthonomal_basis ] = quiz.data
        ins = linear_algebra.InnerSpace(domain='aRn')
        incorrect_basis = _vecs
        if sympy.Matrix(incorrect_basis) != sympy.Matrix(_orthonomal_basis):
            ans['feedback'] = '直交化の次に行う正規化は，ノルムを1にする操作です。最初のベクトルに戻さないでください。'
            ans['data'] = incorrect_basis
            answers.append(dict(ans))
        incorrect_basis = _orthogonal_basis
        if sympy.Matrix(incorrect_basis) != sympy.Matrix(_orthonomal_basis):
            ans['feedback'] = '直交化の次に行う正規化は，ノルムを1にする操作です。そのままではいけません。'
            ans['data'] = incorrect_basis
            answers.append(dict(ans))
        another_ins = linear_algebra.InnerSpace(domain='Rn')
        incorrect_another_basis = another_ins.normalize(_orthogonal_basis)
        if sympy.Matrix(incorrect_another_basis) != sympy.Matrix(_orthonomal_basis):
            ans['feedback'] = '内積の定義に注意してください。標準内積ではありません。'
            ans['data'] = incorrect_another_basis
            answers.append(dict(ans))
        _truefalse = [True for i in range(_dimension - 1)] + [False]
        _truefalse = random.sample(_truefalse, _dimension)        
        incorrect_basis = [_orthonomal_basis[i] if _truefalse[i] else _orthogonal_basis[i] for i in range(_dimension)]
        if sympy.Matrix(incorrect_basis) != sympy.Matrix(_orthonomal_basis):
            ans['feedback'] = '全てのベクトルを正規化してください。'
            ans['data'] = incorrect_basis
            answers.append(dict(ans))      
        incorrect_basis = [incorrect_another_basis[i] if _truefalse[i] else _orthogonal_basis[i] for i in range(_dimension)]
        if sympy.Matrix(incorrect_basis) != sympy.Matrix(_orthonomal_basis):
            ans['feedback'] = '内積の定義に注意してください。標準内積ではありません。'
            ans['data'] = incorrect_basis
            answers.append(dict(ans)) 
        _truefalse = random.sample(_truefalse, _dimension)                
        incorrect_basis = [_orthonomal_basis[i] if _truefalse[i] else _orthogonal_basis[i] for i in range(_dimension)]
        if sympy.Matrix(incorrect_basis) != sympy.Matrix(_orthonomal_basis):
            ans['feedback'] = '全てのベクトルを正規化してください。'
            ans['data'] = incorrect_basis
            answers.append(dict(ans))      
        incorrect_basis = [incorrect_another_basis[i] if _truefalse[i] else _orthogonal_basis[i] for i in range(_dimension)]
        if sympy.Matrix(incorrect_basis) != sympy.Matrix(_orthonomal_basis):
            ans['feedback'] = '内積の定義に注意してください。標準内積ではありません。'
            ans['data'] = incorrect_basis
            answers.append(dict(ans))    
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_dimension, _vecs, _orthogonal_basis, _orthonomal_basis ] = quiz.data
        ins = linear_algebra.InnerSpace(domain='aRn')
        _text = r'ベクトル空間\(\mathbb{R}^{' + str(_dimension) + r'}\)の内積を'
        _text += ins.str_inner_product(is_latex_closure=True)
        _text += r'とします。このとき，次のベクトル'
        for i in range(_dimension):
            _text += r'\(\vec{w_{' + str(i+1) + r'}}\)'
            if i < _dimension - 1:
                _text += r','
        _text += r'を，グラムシュミットの方法で正規直交化することを考えます。'
        _text += r'ひとまず，直交化部分のみ（正規化を含まない）を行ったところ，ベクトル'
        for i in range(_dimension):
            _text += r'\(\vec{v_{' + str(i+1) + r'}}\)'
            if i < _dimension - 1:
                _text += r','
        _text += r'が得られました。次に正規化（正規直交化する残りの手順）を行った場合に得られるベクトルとしてもっとも適切なものを選択してください。<br />'
        _text += ins.str_vectors(_vecs, is_latex_closure=True, variable=r'w') + r'<br />'
        _text += ins.str_vectors(_orthogonal_basis, is_latex_closure=True, variable=r'v')
        return _text
    def answer_text(self, ans):
        if isinstance(ans['data'], str):
            _text = ans['data']
        else:
            ins = linear_algebra.InnerSpace(domain='aRn')
            _text = ins.str_vectors(ans['data'], is_latex_closure=True, variable=None)
        return _text


# In[110]:


if __name__ == "__main__":
    q = normalization_process_in_Rn_with_some_inner_product()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=2)


# In[111]:


if __name__ == "__main__":
    pass
    #qz.save('normalization_process_in_Rn_with_some_inner_product.xml')


# ## some inner product and norm in R[x]

# In[114]:


class some_inner_product_with_norm_in_Rx(core.Question):
    name = 'ある内積とそのノルム（多項式のベクトル空間）'
    def __init__(self, emin=-3, emax=3, dmin=2, dmax=2):
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 生成するベクトルの次元の範囲
        self.dim_min = dmin
        self.dim_max = dmax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='多項式のある内積とそのノルム', quiz_number=_quiz_number)
        ins = linear_algebra.InnerSpace(domain='Rx')
        _dimension = random.randint(self.dim_min, self.dim_max)
        _vectA = [random.randint(self.elem_min, self.elem_max) for i in range(_dimension)]
        _vectB = [random.randint(self.elem_min, self.elem_max) for i in range(_dimension)]
        quiz.quiz_identifier = hash(str(_vectA) + str(_vectB))
        # 正答の選択肢の生成
        _inner_product = ins.inner_product(_vectA, _vectB)
        _normA = ins.norm(_vectA)
        _normB = ins.norm(_vectB)
        quiz.data = [_dimension, _vectA, _vectB, _inner_product, _normA, _normB]
        ans = { 'fraction': 100, 'data': [_inner_product, _normA, _normB] }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_dimension, _vectA, _vectB, _inner_product, _normA, _normB] = quiz.data
        ins = linear_algebra.InnerSpace(domain='Rx')
        ans['feedback'] = '今回の内積は，きちんと内積の条件を満たしています。'
        ans['data'] = [r'この内積はこのベクトル空間では内積の条件を満たさず，定義されない。']
        answers.append(dict(ans))
        incorrect_inner_product = -_inner_product
        incorrect_normA = _normA
        incorrect_normB = _normB
        if incorrect_inner_product != _inner_product:
            ans['feedback'] = '与えられた内積の定義に基づいて計算し直しましょう。'
            ans['data'] = [incorrect_inner_product, incorrect_normA, incorrect_normB]
            answers.append(dict(ans))
        incorrect_inner_product = _inner_product
        incorrect_normA = ins.inner_product(_vectA, _vectA)
        incorrect_normB = ins.inner_product(_vectB, _vectB)
        if incorrect_inner_product != _inner_product or incorrect_normA != _normA or incorrect_normB != _normB:
            ans['feedback'] = 'ノルムは，自分自身との内積の二乗根です。計算し直しましょう。'
            ans['data'] = [incorrect_inner_product, incorrect_normA, incorrect_normB]
            answers.append(dict(ans))
        incorrect_inner_product = -_inner_product
        if incorrect_inner_product != _inner_product or incorrect_normA != _normA or incorrect_normB != _normB:
            ans['feedback'] = 'ノルムは，自分自身との内積の二乗根です。計算し直しましょう。'
            ans['data'] = [incorrect_inner_product, incorrect_normA, incorrect_normB]
            answers.append(dict(ans))
        incorrect_inner_product = sum(_vectA) + sum(_vectB)
        incorrect_normA = abs(sum(_vectA))
        incorrect_normB = abs(sum(_vectB))
        if incorrect_inner_product != _inner_product or incorrect_normA != _normA or incorrect_normB != _normB:
            ans['feedback'] = '与えられた内積の定義に基づいて計算し直しましょう。'
            ans['data'] = [incorrect_inner_product, incorrect_normA, incorrect_normB]
            answers.append(dict(ans))
        incorrect_inner_product = -_inner_product
        if incorrect_inner_product != _inner_product or incorrect_normA != _normA or incorrect_normB != _normB:
            ans['feedback'] = '与えられた内積の定義に基づいて計算し直しましょう。'
            ans['data'] = [incorrect_inner_product, incorrect_normA, incorrect_normB]
            answers.append(dict(ans))
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_dimension, _vectA, _vectB, _inner_product, _normA, _normB] = quiz.data
        ins = linear_algebra.InnerSpace(domain='Rx')
        _text = r'ベクトル空間' + ins.str_domain(is_latex_closure=True) + r'の内積を'
        _text += ins.str_inner_product(is_latex_closure=True)
        _text += r'とする。このとき，次の2つのベクトルの内積とそれぞれのノルムとして適切なものを選択してください。<br />'
        _text += ins.str_vectors([_vectA, _vectB])
        return _text
    def answer_text(self, ans):
        if len(ans['data']) > 1:
            [_inner_product, _normA, _normB] = ans['data']
            _text = r'内積: \(' + sympy.latex(_inner_product) + r'\), '
            _text += r'ノルム: \(' + sympy.latex(_normA) + r',\;' + sympy.latex(_normB) + r'\)'
        else:
            _text = ans['data'][0]
        return _text


# In[115]:


if __name__ == "__main__":
    q = some_inner_product_with_norm_in_Rx()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[116]:


if __name__ == "__main__":
    pass
    #qz.save('some_inner_product_with_norm_in_Rx.xml')


# ## orthogonalization of Gram-Schmidt process in R[x] with some inner product

# In[145]:


class orthogonalization_process_in_Rx_with_some_inner_product(core.Question):
    name = 'グラムシュミットの直交化部分の計算（多項式のベクトル空間，ある内積）'
    def __init__(self, emin=-2, emax=2, dmin=2, dmax=3, zratio=0.5):
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 生成するベクトルの次元の範囲
        self.dim_min = dmin
        self.dim_max = dmax
        # 要素を零とする割合
        self.zero_ratio = zratio
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='直交化部分の計算（R[x]，ある内積）', quiz_number=_quiz_number)
        ins = linear_algebra.InnerSpace(domain='Rx')
        _dimension = random.randint(self.dim_min, self.dim_max)
        _orthonomal_basis = [[1]]
        _mat = sympy.Matrix([[0]])
        while _mat.rank() < _dimension or max([abs(e) for e in linear_algebra.flatten_list(_orthonomal_basis)]) == 1:
            _vecs = [[random.randint(self.elem_min, self.elem_max) if random.random() > self.zero_ratio else 0 for i in range(_dimension)] for j in range(_dimension)]
            _orthonomal_basis = ins.orthonomalize(_vecs)
            _mat = sympy.Matrix(_vecs)
        # 正答の選択肢の生成
        _orthogonal_basis = ins.orthogonalize(_vecs)
        quiz.quiz_identifier = hash(str(_orthogonal_basis))
        quiz.data = [_dimension, _vecs, _orthogonal_basis, _orthonomal_basis ]
        ans = { 'fraction': 100, 'data': _orthogonal_basis[-1] }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_dimension, _vecs, _orthogonal_basis, _orthonomal_basis ] = quiz.data
        ins = linear_algebra.InnerSpace(domain='Rx')
        incorrect_vec = _vecs[-1]
        if incorrect_vec != _orthogonal_basis[-1]:
            ans['feedback'] = '直交化の手順を確認しましょう。正射影分を取り除く必要があります。'
            ans['data'] = incorrect_vec
            answers.append(dict(ans))
        another_ins = linear_algebra.InnerSpace(domain='Rn')
        incorrect_vec = _vecs[-1]
        for v in _orthogonal_basis[:-1]:
            _mu = another_ins.inner_product(v, _vecs[-1]) * sympy.Pow(another_ins.inner_product(v, v), -1)
            incorrect_vec = [incorrect_vec[j] - _mu*v[j] for j in range(len(v))]
        if incorrect_vec != _orthogonal_basis[-1]:
            ans['feedback'] = '内積の定義に注意してください。標準内積ではありません。'
            ans['data'] = incorrect_vec
            answers.append(dict(ans))
        if _dimension > 2:
            incorrect_vec = _vecs[-1]
            for v in _orthogonal_basis[:-2]:
                _mu = ins.inner_product(v, _vecs[-1]) * sympy.Pow(ins.inner_product(v, v), -1)
                incorrect_vec = [incorrect_vec[j] - _mu*v[j] for j in range(len(v))]
            if incorrect_vec != _orthogonal_basis[-1]:
                ans['feedback'] = '直交化の手順を確認しましょう。既に直交化済みのベクトル毎に正射影分を取り除く必要があります。'
                ans['data'] = incorrect_vec
                answers.append(dict(ans))
        incorrect_vec = _vecs[-1]
        for v in _orthogonal_basis[:-1]:
            _mu = ins.inner_product(v, _vecs[-1])# * sympy.Pow(ins.inner_product(v, v), -1)
            incorrect_vec = [incorrect_vec[j] - _mu*v[j] for j in range(len(v))]
        if incorrect_vec != _orthogonal_basis[-1]:
            ans['feedback'] = '直交化の手順を確認しましょう。正規化前なので正射影の大きさは内積2つの有理式になっているはずです。'
            ans['data'] = incorrect_vec
            answers.append(dict(ans))
        incorrect_vec = _vecs[-1]
        for v in _orthogonal_basis[:-1]:
            if ins.inner_product(v,  _vecs[-1]) == 0:
                continue
            _mu = ins.inner_product(v, v) * sympy.Pow(ins.inner_product(v,  _vecs[-1]), -1)
            incorrect_vec = [incorrect_vec[j] - _mu*v[j] for j in range(len(v))]
        if incorrect_vec != _orthogonal_basis[-1]:
            ans['feedback'] = '直交化の手順を確認しましょう。正射影の大きさの2つの内積の役割が逆転しています。'
            ans['data'] = incorrect_vec
            answers.append(dict(ans))
        incorrect_vec = _orthonomal_basis[-1]
        if incorrect_vec != _orthogonal_basis[-1]:
            ans['feedback'] = '直交化のみを行った結果を選んでください。'
            ans['data'] = incorrect_vec
            answers.append(dict(ans))
        answers = common.answer_union(answers)
        for k in range(size - len(answers)):
            incorrect_vec = [_orthogonal_basis[-1][i] + random.choice([-1,0,1])  for i in range(_dimension)]
            if incorrect_vec != _orthogonal_basis[-1]:
                ans['feedback'] = '計算を丁寧に行いましょう。'
                ans['data'] = incorrect_vec
                answers.append(dict(ans))
            answers = common.answer_union(answers)            
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_dimension, _vecs, _orthogonal_basis, _orthonomal_basis ] = quiz.data
        ins = linear_algebra.InnerSpace(domain='Rx')
        _text = r'ベクトル空間\(\mathbb{R}^{' + str(_dimension) + r'}\)の内積を'
        _text += ins.str_inner_product(is_latex_closure=True)
        _text += r'とします。このとき，次のベクトル'
        for i in range(_dimension):
            _text += r'\(\vec{w_{' + str(i+1) + r'}}\)'
            if i < _dimension - 1:
                _text += r','
        _text += r'を，グラムシュミットの方法で正規直交化することを考えます。'
        _text += r'ひとまず，直交化部分のみ（正規化を含まない）を途中まで行ったところ，ベクトル'
        for i in range(_dimension - 1):
            _text += r'\(\vec{v_{' + str(i+1) + r'}}\)'
            if i < _dimension - 2:
                _text += r','
        _text += r'が得られました。次の直交ベクトルを求める手順を行った場合に得られるベクトルとしてもっとも適切なものを選択してください。<br />'
        _text += ins.str_vectors(_vecs, is_latex_closure=True, variable=r'w') + r'<br />'
        _text += ins.str_vectors(_orthogonal_basis[:-1], is_latex_closure=True, variable=r'v')
        return _text
    def answer_text(self, ans):
        if isinstance(ans['data'], str):
            _text = ans['data']
        else:
            ins = linear_algebra.InnerSpace(domain='Rx')
            _text = ins.str_vectors([ans['data']], is_latex_closure=True, variable=None)
        return _text


# In[146]:


if __name__ == "__main__":
    q = orthogonalization_process_in_Rx_with_some_inner_product()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=2)


# In[147]:


if __name__ == "__main__":
    pass
    #qz.save('orthogonalization_process_in_Rx_with_some_inner_product.xml')


# ## normalization of Gram-Schmidt process in R[x] with some inner product

# In[142]:


class normalization_process_in_Rx_with_some_inner_product(core.Question):
    name = 'グラムシュミットの正規化部分の計算（多項式のベクトル空間，ある内積）'
    def __init__(self, emin=-3, emax=3, dmin=2, dmax=2, zratio=0.5):
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 生成するベクトルの次元の範囲
        self.dim_min = dmin
        self.dim_max = dmax
        # 要素を零とする割合
        self.zero_ratio = zratio
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='正規化部分の計算（R[x]，ある内積）', quiz_number=_quiz_number)
        ins = linear_algebra.InnerSpace(domain='Rx')
        _dimension = random.randint(self.dim_min, self.dim_max)
        _orthonomal_basis = [[1]]
        _mat = sympy.Matrix([[0]])
        while _mat.rank() < _dimension or max([abs(e) for e in linear_algebra.flatten_list(_orthonomal_basis)]) == 1:
            _vecs = [[random.randint(self.elem_min, self.elem_max) if random.random() > self.zero_ratio else 0 for i in range(_dimension)] for j in range(_dimension)]
            _orthonomal_basis = ins.orthonomalize(_vecs)
            _mat = sympy.Matrix(_vecs)
        # 正答の選択肢の生成
        _orthogonal_basis = ins.orthogonalize(_vecs)
        quiz.quiz_identifier = hash(str(_orthogonal_basis))
        quiz.data = [_dimension, _vecs, _orthogonal_basis, _orthonomal_basis ]
        ans = { 'fraction': 100, 'data': _orthonomal_basis }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_dimension, _vecs, _orthogonal_basis, _orthonomal_basis ] = quiz.data
        ins = linear_algebra.InnerSpace(domain='Rx')
        incorrect_basis = _vecs
        if sympy.Matrix(incorrect_basis) != sympy.Matrix(_orthonomal_basis):
            ans['feedback'] = '直交化の次に行う正規化は，ノルムを1にする操作です。最初のベクトルに戻さないでください。'
            ans['data'] = incorrect_basis
            answers.append(dict(ans))
        incorrect_basis = _orthogonal_basis
        if sympy.Matrix(incorrect_basis) != sympy.Matrix(_orthonomal_basis):
            ans['feedback'] = '直交化の次に行う正規化は，ノルムを1にする操作です。そのままではいけません。'
            ans['data'] = incorrect_basis
            answers.append(dict(ans))
        another_ins = linear_algebra.InnerSpace(domain='Rn')
        incorrect_another_basis = another_ins.normalize(_orthogonal_basis)
        if sympy.Matrix(incorrect_another_basis) != sympy.Matrix(_orthonomal_basis):
            ans['feedback'] = '内積の定義に注意してください。標準内積ではありません。'
            ans['data'] = incorrect_another_basis
            answers.append(dict(ans))
        _truefalse = [True for i in range(_dimension - 1)] + [False]
        _truefalse = random.sample(_truefalse, _dimension)        
        incorrect_basis = [_orthonomal_basis[i] if _truefalse[i] else _orthogonal_basis[i] for i in range(_dimension)]
        if sympy.Matrix(incorrect_basis) != sympy.Matrix(_orthonomal_basis):
            ans['feedback'] = '全てのベクトルを正規化してください。'
            ans['data'] = incorrect_basis
            answers.append(dict(ans))      
        incorrect_basis = [incorrect_another_basis[i] if _truefalse[i] else _orthogonal_basis[i] for i in range(_dimension)]
        if sympy.Matrix(incorrect_basis) != sympy.Matrix(_orthonomal_basis):
            ans['feedback'] = '内積の定義に注意してください。標準内積ではありません。'
            ans['data'] = incorrect_basis
            answers.append(dict(ans)) 
        _truefalse = random.sample(_truefalse, _dimension)                
        incorrect_basis = [_orthonomal_basis[i] if _truefalse[i] else _orthogonal_basis[i] for i in range(_dimension)]
        if sympy.Matrix(incorrect_basis) != sympy.Matrix(_orthonomal_basis):
            ans['feedback'] = '全てのベクトルを正規化してください。'
            ans['data'] = incorrect_basis
            answers.append(dict(ans))      
        incorrect_basis = [incorrect_another_basis[i] if _truefalse[i] else _orthogonal_basis[i] for i in range(_dimension)]
        if sympy.Matrix(incorrect_basis) != sympy.Matrix(_orthonomal_basis):
            ans['feedback'] = '内積の定義に注意してください。標準内積ではありません。'
            ans['data'] = incorrect_basis
            answers.append(dict(ans))    
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_dimension, _vecs, _orthogonal_basis, _orthonomal_basis ] = quiz.data
        ins = linear_algebra.InnerSpace(domain='Rx')
        _text = r'ベクトル空間\(\mathbb{R}^{' + str(_dimension) + r'}\)の内積を'
        _text += ins.str_inner_product(is_latex_closure=True)
        _text += r'とします。このとき，次のベクトル'
        for i in range(_dimension):
            _text += r'\(\vec{w_{' + str(i+1) + r'}}\)'
            if i < _dimension - 1:
                _text += r','
        _text += r'を，グラムシュミットの方法で正規直交化することを考えます。'
        _text += r'ひとまず，直交化部分のみ（正規化を含まない）を行ったところ，ベクトル'
        for i in range(_dimension):
            _text += r'\(\vec{v_{' + str(i+1) + r'}}\)'
            if i < _dimension - 1:
                _text += r','
        _text += r'が得られました。次に正規化（正規直交化する残りの手順）を行った場合に得られるベクトルとしてもっとも適切なものを選択してください。<br />'
        _text += ins.str_vectors(_vecs, is_latex_closure=True, variable=r'w') + r'<br />'
        _text += ins.str_vectors(_orthogonal_basis, is_latex_closure=True, variable=r'v')
        return _text
    def answer_text(self, ans):
        if isinstance(ans['data'], str):
            _text = ans['data']
        else:
            ins = linear_algebra.InnerSpace(domain='Rx')
            _text = ins.str_vectors(ans['data'], is_latex_closure=True, variable=None)
        return _text


# In[143]:


if __name__ == "__main__":
    q = normalization_process_in_Rx_with_some_inner_product()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=2)


# In[144]:


if __name__ == "__main__":
    pass
    #qz.save('normalization_process_in_Rx_with_some_inner_product.xml')


# ## select orthogonal matrix

# In[151]:


class select_orthogonal_matrix(core.Question):
    name = '直交行列の選択'
    def __init__(self, emin=-2, emax=2, dmin=2, dmax=3, zratio=0.5):
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 生成するベクトルの次元の範囲
        self.dim_min = dmin
        self.dim_max = dmax
        # 要素を零とする割合
        self.zero_ratio = zratio
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='直交行列の選択', quiz_number=_quiz_number)
        ins = linear_algebra.InnerSpace(domain='Rn')
        _dimension = random.randint(self.dim_min, self.dim_max)
        _orthonomal_basis = [[1]]
        _mat = sympy.Matrix([[0]])
        while _mat.rank() < _dimension or max([abs(e) for e in linear_algebra.flatten_list(_orthonomal_basis)]) == 1:
            _vecs = [[random.randint(self.elem_min, self.elem_max) if random.random() > self.zero_ratio else 0 for i in range(_dimension)] for j in range(_dimension)]
            _orthonomal_basis = ins.orthonomalize(_vecs)
            _mat = sympy.Matrix(_vecs)
        # 正答の選択肢の生成
        quiz.quiz_identifier = hash(str(_orthonomal_basis))
        quiz.data = [_dimension, _vecs, _orthonomal_basis]
        ans = { 'fraction': 100, 'data': _orthonomal_basis }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_dimension, _vecs, _orthonomal_basis] = quiz.data
        ins = linear_algebra.InnerSpace(domain='Rn')
        ans['feedback'] = '含まれているので，行ないしは列ベクトルで，直交性と正規性をそれぞれ確認してください。'
        ans['data'] = r'直交行列は含まれていない。'
        answers.append(dict(ans))
        _elems = list(range(0, max(abs(self.elem_min), abs(self.elem_max)) + 1))
        _matE = sympy.eye(_dimension)
        incorrect_basis = ins.orthogonalize(_vecs)
        _mat = sympy.Matrix(incorrect_basis)
        if _mat*_mat.transpose() != _matE or _mat.transpose()*_mat != _matE:
            ans['feedback'] = '直交行列とは，行ないしは列ベクトルで，それぞれが直交かつ正規である必要があります。'
            ans['data'] = incorrect_basis
            answers.append(dict(ans))
        if _dimension > 1:
            incorrect_basis = _vecs[1:]
            _mu = [random.choice([-2,-1,1,2]) for i in range(len(incorrect_basis))]
            incorrect_basis.append([sum([_mu[j]*incorrect_basis[j][i] for j in range(len(incorrect_basis))]) for i in range(_dimension)])
            _mat = sympy.Matrix(incorrect_basis)
            if _mat.rank() < _dimension:
                ans['feedback'] = '直交行列は，まずは正則である必要があります。'
                ans['data'] = incorrect_basis
                answers.append(dict(ans))
            incorrect_basis = ins.orthonomalize(_vecs)[1:]
            _mu = random.choice(_elems + [-e for e in _elems] + [sympy.sqrt(e) for e in _elems] + [-sympy.sqrt(e) for e in _elems])
            _vec = random.choice(incorrect_basis)
            incorrect_basis.append([_mu*e for e in _vec])
            _mat = sympy.Matrix(incorrect_basis)
            if _mat.rank() < _dimension:
                ans['feedback'] = '直交行列は，まずは正則である必要があります。'
                ans['data'] = incorrect_basis
                answers.append(dict(ans))
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        _text = r'直交行列を選択してください。'
        return _text
    def answer_text(self, ans):
        if isinstance(ans['data'], str):
            _text = ans['data']
        else:
            _text = r'\(' + sympy.latex(sympy.Matrix(ans['data']), mat_delim='', mat_str='pmatrix')  + r'\)'
        return _text


# In[152]:


if __name__ == "__main__":
    q = select_orthogonal_matrix()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=2)


# In[153]:


if __name__ == "__main__":
    pass
    #qz.save('select_orthogonal_matrix.xml')


# ## unknown element in orthogonal matrix

# In[163]:


class unknown_element_in_orthogonal_matrix(core.Question):
    name = '直交行列となるパラメータの値の選択'
    def __init__(self, emin=-2, emax=2, dmin=2, dmax=3, zratio=0.5):
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 生成するベクトルの次元の範囲
        self.dim_min = dmin
        self.dim_max = dmax
        # 要素を零とする割合
        self.zero_ratio = zratio
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='直交行列となるパラメータの値', quiz_number=_quiz_number)
        ins = linear_algebra.InnerSpace(domain='Rn')
        _dimension = random.randint(self.dim_min, self.dim_max)
        _orthonomal_basis = [[1]]
        _mat = sympy.Matrix([[0]])
        while _mat.rank() < _dimension or max([abs(e) for e in linear_algebra.flatten_list(_orthonomal_basis)]) == 1:
            _vecs = [[random.randint(self.elem_min, self.elem_max) if random.random() > self.zero_ratio else 0 for i in range(_dimension)] for j in range(_dimension)]
            _orthonomal_basis = ins.orthonomalize(_vecs)
            _mat = sympy.Matrix(_vecs)
        # 正答の選択肢の生成
        quiz.quiz_identifier = hash(str(_orthonomal_basis))
        _unknown_i = random.randint(0,_dimension-1)
        _unknown_j = random.randint(0,_dimension-1)
        quiz.data = [_dimension, _vecs, _orthonomal_basis, _unknown_i, _unknown_j]
        ans = { 'fraction': 100, 'data': _orthonomal_basis[_unknown_i][_unknown_j] }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_dimension, _vecs, _orthonomal_basis, _unknown_i, _unknown_j] = quiz.data
        ins = linear_algebra.InnerSpace(domain='Rn')
        correct_answer = _orthonomal_basis[_unknown_i][_unknown_j]
        _all_elements = set([abs(e) for e in linear_algebra.flatten_list(_orthonomal_basis)] + 
                            [-abs(e) for e in linear_algebra.flatten_list(_orthonomal_basis)] + 
                            list(range(self.elem_min,self.elem_max)))
        for e in _all_elements:
            if sympy.simplify(e - correct_answer) != 0:
                ans['feedback'] = '定義に基づいて，直交性と正規性を満たすように選んでください。'
                ans['data'] = e
                answers.append(dict(ans))
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_dimension, _vecs, _orthonomal_basis, _unknown_i, _unknown_j] = quiz.data
        _text = r'次の行列が直交行列となるようにパラメータ\(\alpha\)の値を定めてください。<br />'
        _alpha = sympy.Symbol('alpha')
        _obasis = copy.deepcopy(_orthonomal_basis)
        _obasis[_unknown_i][_unknown_j] = _alpha
        _text += r'\(' + sympy.latex(sympy.Matrix(_obasis), mat_delim='', mat_str='pmatrix')  + r'\)'
        return _text
    def answer_text(self, ans):
        if isinstance(ans['data'], str):
            _text = ans['data']
        else:
            _text = r'\(' + sympy.latex(ans['data'])  + r'\)'
        return _text


# In[164]:


if __name__ == "__main__":
    q = unknown_element_in_orthogonal_matrix()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=2)


# In[165]:


if __name__ == "__main__":
    pass
    #qz.save('unknown_element_in_orthogonal_matrix.xml')


# ## select symmetric matrix

# In[192]:


class select_symmetric_matrix(core.Question):
    name = '対称行列の選択'
    def __init__(self, emin=-2, emax=2, dmin=2, dmax=3):
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 生成するベクトルの次元の範囲
        self.dim_min = dmin
        self.dim_max = dmax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='対称行列の選択', quiz_number=_quiz_number)
        _dimension = random.randint(self.dim_min, self.dim_max)
        # 正答の選択肢の生成
        quiz.quiz_identifier = hash(str(random.random()))
        _elems = [abs(e) for e in range(self.elem_min, self.elem_max)]
        _elems = list(set(_elems + [-e for e in _elems] + [sympy.sqrt(e) for e in _elems] + [-sympy.sqrt(e) for e in _elems]))
        quiz.data = [_dimension, _elems]
        symmetric_basis = [[0 for i in range(_dimension)] for j in range(_dimension)]
        for i in range(_dimension):
            for j in range(i,_dimension):
                _e = random.choice(_elems)
                symmetric_basis[i][j] = _e
                symmetric_basis[j][i] = _e
        ans = { 'fraction': 100, 'data': symmetric_basis }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_dimension, _elems] = quiz.data
        for k in range(2):
            incorrect_basis = [[0 for i in range(_dimension)] for j in range(_dimension)]
            for i in range(sympy.ceiling(_dimension/2)):
                for j in range(_dimension):
                    _e = random.choice(_elems)
                    incorrect_basis[i][j] = _e
                    incorrect_basis[_dimension - i -1][j] = _e
            if sympy.Matrix(incorrect_basis) != sympy.Matrix(incorrect_basis).transpose():
                ans['feedback'] = r'対称行列とは，転置操作で不変な行列であり，列や行での対称性ではありません。'
                ans['data'] = incorrect_basis
                answers.append(dict(ans))
            incorrect_basis = [[0 for i in range(_dimension)] for j in range(_dimension)]
            for i in range(sympy.ceiling(_dimension/2)):
                for j in range(_dimension):
                    _e = random.choice(_elems)
                    incorrect_basis[j][i] = _e
                    incorrect_basis[j][_dimension - i -1] = _e
            if sympy.Matrix(incorrect_basis) != sympy.Matrix(incorrect_basis).transpose():
                ans['feedback'] = r'対称行列とは，転置操作で不変な行列であり，列や行での対称性ではありません。'
                ans['data'] = incorrect_basis
                answers.append(dict(ans))
            ins = linear_algebra.InnerSpace(domain='Rn')
            incorrect_basis = [[1]]
            while sympy.Matrix(incorrect_basis) == sympy.Matrix(incorrect_basis).transpose() or sympy.Matrix(_vecs).rank() < _dimension:
                _vecs = [[random.randint(self.elem_min, self.elem_max) if random.random() > 0.5 else 0 for i in range(_dimension)] for j in range(_dimension)]
                incorrect_basis = ins.orthonomalize(_vecs)
            if sympy.Matrix(incorrect_basis) != sympy.Matrix(incorrect_basis).transpose():
                ans['feedback'] = r'対称行列とは，転置操作で不変な行列であり，直交行列とは少し異なります。'
                ans['data'] = incorrect_basis
                answers.append(dict(ans))
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        _text = r'対称行列を選択してください。'
        return _text
    def answer_text(self, ans):
        if isinstance(ans['data'], str):
            _text = ans['data']
        else:
            _text = r'\(' + sympy.latex(sympy.Matrix(ans['data']), mat_delim='', mat_str='pmatrix')  + r'\)'
        return _text


# In[193]:


if __name__ == "__main__":
    q = select_symmetric_matrix()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=2)


# In[194]:


if __name__ == "__main__":
    pass
    #qz.save('select_symmetric_matrix.xml')


# ## select unitary matrix

# In[6]:


class select_unitary_matrix(core.Question):
    name = 'ユニタリー行列の選択'
    def __init__(self, emin=-1, emax=1, dmin=2, dmax=2, zratio=0.25):
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 生成するベクトルの次元の範囲
        self.dim_min = dmin
        self.dim_max = dmax
        # 初期に零にする割合
        self.zero_ratio = zratio
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='ユニタリー行列の選択', quiz_number=_quiz_number)
        _dimension = random.randint(self.dim_min, self.dim_max)
        # 正答の選択肢の生成
        quiz.quiz_identifier = hash(str(random.random()))
        _elems = [abs(e) for e in range(self.elem_min, self.elem_max)]
        _elems = list(set(_elems + [-e for e in _elems] + [sympy.sqrt(e) for e in _elems] + [-sympy.sqrt(e) for e in _elems]))
        ins = linear_algebra.InnerSpace(domain='Cn')
        unitary_basis = [[1]]
        _vecs = [[1]]
        while sympy.Matrix(unitary_basis) == sympy.Matrix(unitary_basis).transpose() or sympy.Matrix(_vecs).rank() < _dimension:
            _vecs = [[random.randint(self.elem_min, self.elem_max) if random.random() > self.zero_ratio else 0 for i in range(_dimension)] for j in range(_dimension)]
            _vecs = [[_vecs[i][j] + sympy.I*random.randint(self.elem_min, self.elem_max) if random.random() > self.zero_ratio else 0 for i in range(_dimension)] for j in range(_dimension)]
            unitary_basis = ins.orthonomalize(_vecs)
        unitary_basis = [[sympy.expand(sympy.simplify(e)) for e in v] for v in unitary_basis]
        if self.is_not_unitary(sympy.Matrix(unitary_basis)):
            print(_vecs)
        _elems = list(set(_elems + linear_algebra.flatten_list(unitary_basis)))
        quiz.data = [_dimension, _elems]
        ans = { 'fraction': 100, 'data': unitary_basis }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    @classmethod
    def is_not_unitary(cls, mat):
        wa = sum([abs(e) for e in linear_algebra.flatten_list(sympy.matrix2numpy(mat*mat.adjoint()).tolist())])
        wa_numeric = sympy.re(sympy.N(wa))
        wa_upper = mat.shape[0] + 0.01
        wa_lower = mat.shape[0] - 0.01
        if wa_lower < wa_numeric and wa_numeric < wa_upper:
            return False
        else:
            return True
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_dimension, _elems] = quiz.data
        for k in range(2):
            incorrect_basis = [[random.choice(_elems) for i in range(_dimension)] for j in range(_dimension)]
            if self.is_not_unitary(sympy.Matrix(incorrect_basis)):
                ans['feedback'] = r'ユニタリー行列とは，複素共役転置行列（随伴行列）との積が単位行列となる行列です。'
                ans['data'] = incorrect_basis
                answers.append(dict(ans))
            incorrect_basis = [[0 for i in range(_dimension)] for j in range(_dimension)]
            for i in range(_dimension):
                for j in range(i,_dimension):
                    _e = random.choice(_elems)
                    incorrect_basis[i][j] = _e
                    incorrect_basis[j][i] = _e
            if self.is_not_unitary(sympy.Matrix(incorrect_basis)):
                ans['feedback'] = r'ユニタリー行列とは，複素共役転置行列（随伴行列）との積が単位行列となる行列であり，対称行列ではありません。'
                ans['data'] = incorrect_basis
                answers.append(dict(ans))
            incorrect_basis = [[0 for i in range(_dimension)] for j in range(_dimension)]
            for i in range(sympy.ceiling(_dimension/2)):
                for j in range(_dimension):
                    _e = random.choice(_elems)
                    incorrect_basis[i][j] = _e
                    incorrect_basis[_dimension - i -1][j] = _e
            if self.is_not_unitary(sympy.Matrix(incorrect_basis)):
                ans['feedback'] = r'ユニタリー行列とは，複素共役転置行列（随伴行列）との積が単位行列となる行列であり，列や行での対称性ではありません。'
                ans['data'] = incorrect_basis
                answers.append(dict(ans))
            incorrect_basis = [[0 for i in range(_dimension)] for j in range(_dimension)]
            for i in range(sympy.ceiling(_dimension/2)):
                for j in range(_dimension):
                    _e = random.choice(_elems)
                    incorrect_basis[j][i] = _e
                    incorrect_basis[j][_dimension - i -1] = _e
            if self.is_not_unitary(sympy.Matrix(incorrect_basis)):
                ans['feedback'] = r'ユニタリー行列とは，複素共役転置行列（随伴行列）との積が単位行列となる行列であり，列や行での対称性ではありません。'
                ans['data'] = incorrect_basis
                answers.append(dict(ans))
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        _text = r'ユニタリー行列を選択してください。'
        return _text
    def answer_text(self, ans):
        if isinstance(ans['data'], str):
            _text = ans['data']
        else:
            _text = r'\(' + sympy.latex(sympy.Matrix(ans['data']), mat_delim='', mat_str='pmatrix')  + r'\)'
        return _text


# In[7]:


if __name__ == "__main__":
    q = select_unitary_matrix()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[8]:


if __name__ == "__main__":
    pass
    #qz.save('select_unitary_matrix.xml')


# ## select Hermitian matrix

# In[34]:


class select_Hermitian_matrix(core.Question):
    name = 'エルミート行列の選択'
    def __init__(self, emin=-3, emax=3, dmin=2, dmax=3):
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 生成するベクトルの次元の範囲
        self.dim_min = dmin
        self.dim_max = dmax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='エルミート行列の選択', quiz_number=_quiz_number)
        _dimension = random.randint(self.dim_min, self.dim_max)
        # 正答の選択肢の生成
        quiz.quiz_identifier = hash(str(random.random()))
        _elems = [abs(e) for e in range(self.elem_min, self.elem_max)]
        _elems = list(set(_elems + [-e for e in _elems] + [sympy.sqrt(e) for e in _elems] + [-sympy.sqrt(e) for e in _elems]))
        _elems = list(set(linear_algebra.flatten_list([[_elems[i] + _elems[j]*sympy.I for i in range(len(_elems))] for j in range(len(_elems))])))
        quiz.data = [_dimension, _elems]
        Hermitian_basis = [[0 for i in range(_dimension)] for j in range(_dimension)]
        for i in range(_dimension):
            for j in range(i,_dimension):
                _e = random.choice(_elems)
                if i == j:
                    Hermitian_basis[i][j] = sympy.re(_e)
                else:
                    Hermitian_basis[i][j] = _e
                    Hermitian_basis[j][i] = sympy.conjugate(_e)
        ans = { 'fraction': 100, 'data': Hermitian_basis }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_dimension, _elems] = quiz.data
        for k in range(2):
            incorrect_basis = [[random.choice(_elems) for i in range(_dimension)] for j in range(_dimension)]
            for i in range(_dimension):
                incorrect_basis[i][i] = sympy.re(incorrect_basis[i][i])
            if sympy.Matrix(incorrect_basis) != sympy.Matrix(incorrect_basis).adjoint():
                ans['feedback'] = r'エルミート行列とは，複素共役転置操作（随伴行列）で不変な行列です。'
                ans['data'] = incorrect_basis
                answers.append(dict(ans))
            incorrect_basis = [[0 for i in range(_dimension)] for j in range(_dimension)]
            for i in range(_dimension):
                for j in range(i,_dimension):
                    _e = random.choice(_elems)
                    if i == j:
                        incorrect_basis[i][j] = sympy.re(_e)
                    else:
                        incorrect_basis[i][j] = _e
                        incorrect_basis[j][i] = _e
            if sympy.Matrix(incorrect_basis) != sympy.Matrix(incorrect_basis).adjoint():
                ans['feedback'] = r'エルミート行列とは，複素共役転置操作（随伴行列）で不変な行列であり，対称行列ではありません。'
                ans['data'] = incorrect_basis
                answers.append(dict(ans))
            incorrect_basis = [[0 for i in range(_dimension)] for j in range(_dimension)]
            for i in range(sympy.ceiling(_dimension/2)):
                for j in range(_dimension):
                    _e = random.choice(_elems)
                    incorrect_basis[i][j] = _e
                    incorrect_basis[_dimension - i -1][j] = _e
            if sympy.Matrix(incorrect_basis) != sympy.Matrix(incorrect_basis).adjoint():
                ans['feedback'] = r'エルミート行列とは，複素共役転置操作（随伴行列）で不変な行列であり，列や行での対称性ではありません。'
                ans['data'] = incorrect_basis
                answers.append(dict(ans))
            incorrect_basis = [[0 for i in range(_dimension)] for j in range(_dimension)]
            for i in range(sympy.ceiling(_dimension/2)):
                for j in range(_dimension):
                    _e = random.choice(_elems)
                    incorrect_basis[j][i] = _e
                    incorrect_basis[j][_dimension - i -1] = _e
            if sympy.Matrix(incorrect_basis) != sympy.Matrix(incorrect_basis).adjoint():
                ans['feedback'] = r'エルミート行列とは，複素共役転置操作（随伴行列）で不変な行列であり，列や行での対称性ではありません。'
                ans['data'] = incorrect_basis
                answers.append(dict(ans))
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        _text = r'エルミート行列を選択してください。'
        return _text
    def answer_text(self, ans):
        if isinstance(ans['data'], str):
            _text = ans['data']
        else:
            _text = r'\(' + sympy.latex(sympy.Matrix(ans['data']), mat_delim='', mat_str='pmatrix')  + r'\)'
        return _text


# In[35]:


if __name__ == "__main__":
    q = select_Hermitian_matrix()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[36]:


if __name__ == "__main__":
    pass
    #qz.save('select_Hermitian_matrix.xml')


# ## select normal matrix

# In[9]:


class select_normal_matrix(core.Question):
    name = '正規行列の選択'
    def __init__(self, emin=-3, emax=3, dmin=2, dmax=3):
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 生成するベクトルの次元の範囲
        self.dim_min = dmin
        self.dim_max = dmax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='正規行列の選択', quiz_number=_quiz_number)
        # 正答の選択肢の生成
        quiz.quiz_identifier = hash(str(random.random()))
        _elems = [abs(e) for e in range(self.elem_min, self.elem_max)]
        _elems = list(set(_elems + [-e for e in _elems] + [sympy.sqrt(e) for e in _elems] + [-sympy.sqrt(e) for e in _elems]))
        _elems = list(set(linear_algebra.flatten_list([[_elems[i] + _elems[j]*sympy.I for i in range(len(_elems))] for j in range(len(_elems))])))
        _type = random.choice(['orthonormal', 'unitary', 'symmetric', 'Hermitian'])
        if _type == 'orthonormal':
            _dimension = random.randint(self.dim_min, self.dim_max)
            ins = linear_algebra.InnerSpace(domain='Rn')
            correct_basis = [[1]]
            _mat = sympy.Matrix([[0]])
            while _mat.rank() < _dimension or max([abs(e) for e in linear_algebra.flatten_list(correct_basis)]) == 1:
                _vecs = [[random.randint(self.elem_min+1, self.elem_max-1) if random.random() > 0.5 else 0 for i in range(_dimension)] for j in range(_dimension)]
                correct_basis = ins.orthonomalize(_vecs)
                _mat = sympy.Matrix(_vecs)
        elif _type == 'unitary':
            _dimension = random.randint(self.dim_min, self.dim_max-1)
            ins = linear_algebra.InnerSpace(domain='Cn')
            correct_basis = [[1]]
            _vecs = [[1]]
            while sympy.Matrix(correct_basis) == sympy.Matrix(correct_basis).transpose() or sympy.Matrix(_vecs).rank() < _dimension:
                _vecs = [[random.randint(self.elem_min+2, self.elem_max-2) if random.random() > 0.25 else 0 for i in range(_dimension)] for j in range(_dimension)]
                _vecs = [[_vecs[i][j] + sympy.I*random.randint(self.elem_min+2, self.elem_max-2) if random.random() > 0.25 else 0 for i in range(_dimension)] for j in range(_dimension)]
                correct_basis = ins.orthonomalize(_vecs)
            correct_basis = [[sympy.expand(sympy.simplify(e)) for e in v] for v in correct_basis]
        elif _type == 'symmetric':
            _dimension = random.randint(self.dim_min, self.dim_max)
            correct_basis = [[0 for i in range(_dimension)] for j in range(_dimension)]
            for i in range(_dimension):
                for j in range(i,_dimension):
                    _e = random.choice(_elems)
                    correct_basis[i][j] = sympy.re(_e)
                    correct_basis[j][i] = sympy.re(_e)
        else:
            _dimension = random.randint(self.dim_min, self.dim_max)
            correct_basis = [[0 for i in range(_dimension)] for j in range(_dimension)]
            for i in range(_dimension):
                for j in range(i,_dimension):
                    _e = random.choice(_elems)
                    if i == j:
                        correct_basis[i][j] = sympy.re(_e)
                    else:
                        correct_basis[i][j] = _e
                        correct_basis[j][i] = sympy.conjugate(_e)
        if self.is_not_normal(sympy.Matrix(correct_basis)):
            print(correct_basis)
        _elems = list(set(_elems + linear_algebra.flatten_list(correct_basis)))
        quiz.data = [_dimension, _elems]
        ans = { 'fraction': 100, 'data': correct_basis }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    @classmethod
    def is_not_normal(cls, mat):
        sa = sum([abs(e) for e in linear_algebra.flatten_list(sympy.matrix2numpy(mat*mat.adjoint()-mat.adjoint()*mat).tolist())])
        sa_numeric = sympy.re(sympy.N(sa))
        sa_upper = 0.01
        sa_lower = - 0.01
        if sa_lower < sa_numeric and sa_numeric < sa_upper:
            return False
        else:
            return True
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_dimension, _elems] = quiz.data
        for k in range(2):
            incorrect_basis = [[random.choice(_elems) for i in range(_dimension)] for j in range(_dimension)]
            for i in range(_dimension):
                incorrect_basis[i][i] = sympy.re(incorrect_basis[i][i])
            if self.is_not_normal(sympy.Matrix(incorrect_basis)):
                ans['feedback'] = r'正規行列とは，複素共役転置行列（随伴行列）との積が可換な行列です。'
                ans['data'] = incorrect_basis
                answers.append(dict(ans))
            incorrect_basis = [[0 for i in range(_dimension)] for j in range(_dimension)]
            for i in range(_dimension):
                for j in range(i,_dimension):
                    _e = random.choice(_elems)
                    if i == j:
                        incorrect_basis[i][j] = sympy.re(_e)
                    else:
                        incorrect_basis[i][j] = _e
                        incorrect_basis[j][i] = _e
            if self.is_not_normal(sympy.Matrix(incorrect_basis)):
                ans['feedback'] = r'正規行列とは，複素共役転置行列（随伴行列）との積が可換な行列であり，非実の対称行列ではありません。'
                ans['data'] = incorrect_basis
                answers.append(dict(ans))
            incorrect_basis = [[0 for i in range(_dimension)] for j in range(_dimension)]
            for i in range(sympy.ceiling(_dimension/2)):
                for j in range(_dimension):
                    _e = random.choice(_elems)
                    incorrect_basis[i][j] = _e
                    incorrect_basis[_dimension - i -1][j] = _e
            if self.is_not_normal(sympy.Matrix(incorrect_basis)):
                ans['feedback'] = r'正規行列とは，複素共役転置行列（随伴行列）との積が可換な行列であり，列や行での対称性ではありません。'
                ans['data'] = incorrect_basis
                answers.append(dict(ans))
            incorrect_basis = [[0 for i in range(_dimension)] for j in range(_dimension)]
            for i in range(sympy.ceiling(_dimension/2)):
                for j in range(_dimension):
                    _e = random.choice(_elems)
                    incorrect_basis[j][i] = _e
                    incorrect_basis[j][_dimension - i -1] = _e
            if self.is_not_normal(sympy.Matrix(incorrect_basis)):
                ans['feedback'] = r'正規行列とは，複素共役転置行列（随伴行列）との積が可換な行列であり，列や行での対称性ではありません。'
                ans['data'] = incorrect_basis
                answers.append(dict(ans))
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        _text = r'正規行列を選択してください。'
        return _text
    def answer_text(self, ans):
        if isinstance(ans['data'], str):
            _text = ans['data']
        else:
            _text = r'\(' + sympy.latex(sympy.Matrix(ans['data']), mat_delim='', mat_str='pmatrix')  + r'\)'
        return _text


# In[10]:


if __name__ == "__main__":
    q = select_normal_matrix()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[11]:


if __name__ == "__main__":
    pass
    #qz.save('select_normal_matrix.xml')


# ## diagonalization of real symmetric matrix

# In[34]:


class diagonalization_of_real_symmetric_matrix(core.Question):
    name = '実対称行列が表現行列となる場合の対角化（固有空間から）'
    def __init__(self, emin=-3, emax=3, dmin=2, dmax=3):
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 生成するベクトルの次元の範囲
        self.dim_min = dmin
        self.dim_max = dmax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='実対称行列の対角化（固有空間から）', quiz_number=_quiz_number)
        _dimension = random.randint(self.dim_min, self.dim_max)
        while True:
            _representation_matrix = [[0 for i in range(_dimension)] for j in range(_dimension)]
            _off_diagonals = 0
            for i in range(_dimension):
                for j in range(i, _dimension):
                    _representation_matrix[i][j] = random.randint(self.elem_min, self.elem_max)
                    _representation_matrix[j][i] = _representation_matrix[i][j]
                    if i != j:
                        _off_diagonals += abs(_representation_matrix[i][j])
            if _off_diagonals == 0:
                continue
            _mat = sympy.Matrix(_representation_matrix)
            if not _mat.is_diagonalizable(reals_only=True):
                continue
            _evs = _mat.eigenvals(multiple=False)
            _is_integer = True
            for ev in _evs.keys():
                if not sympy.sympify(ev).is_integer:
                    _is_integer = False
                    break
#            if not _is_integer:
#                continue
#            for _evct in _mat.eigenvects():
#                for e in linear_algebra.flatten_list([sympy.matrix2numpy(v.transpose()).tolist()[0] for v in _evct[2]]):
#                    if not sympy.sympify(e).is_integer:
#                        _is_integer = False
#                        break
#                if not _is_integer:
#                    break
            if _is_integer:
                break
        es = linear_algebra.EigenSpace()
        es.dimension = _dimension
        es.representation_matrix = _representation_matrix
        es.eigen_space_dimensions = []
        es.eigen_values = []
        es.eigen_vectors = []
        for _evct in _mat.eigenvects():
            es.eigen_values.append(_evct[0])
            es.eigen_space_dimensions.append(_evct[1])
            _evct_list = [sympy.matrix2numpy(v.transpose()).tolist()[0] for v in _evct[2]]
            for i in range(len(_evct_list)):
                _lcm = sympy.lcm([sympy.sympify(e).as_numer_denom()[1] for e in _evct_list[i]])
                _evct_list[i] = [e*_lcm for e in _evct_list[i]]
            es.eigen_vectors.append(_evct_list)
        quiz.quiz_identifier = hash(es)
        # 正答の選択肢の生成
        _evcts = sympy.GramSchmidt([sympy.Matrix(e) for e in linear_algebra.flatten_list(es.eigen_vectors)],True)
        _evcts = [sympy.matrix2numpy(e.transpose()).tolist()[0] for e in _evcts]
        _evs = linear_algebra.flatten_list([[es.eigen_values[i] for j in range(len(es.eigen_vectors[i]))] for i in range(len(es.eigen_values))])
        quiz.data = [es, _evs, _evcts]
        ans = { 'fraction': 100, 'data': [_evs, _evcts] }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def _is_correct(self, matA, evs, evcts):
        _matP = sympy.Matrix(evcts).transpose()
        _matD = sympy.diag(*evs)
        if _matP.inv()*matA*_matP == _matD:
            return True
        else:
            return False
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        es = quiz.data[0]
        correct_evs = quiz.data[1]
        correct_evcts = quiz.data[2]
        matA = sympy.Matrix(es.representation_matrix)
        
        incorrect_evcts = linear_algebra.flatten_list(es.eigen_vectors)
        incorrect_evs = correct_evs
        if matA != sympy.Matrix(incorrect_evcts):
            ans['feedback'] = r'直交行列で対角化してください。'
            ans['data'] = [incorrect_evs, incorrect_evcts]
            answers.append(dict(ans))
            for i in range(len(correct_evs)):
                incorrect_evs = correct_evs[i:] + correct_evs[:i]
                incorrect_evcts = linear_algebra.flatten_list(es.eigen_vectors)
                if matA != sympy.Matrix(incorrect_evcts):
                    ans['feedback'] = r'直交行列で対角化してください。'
                    ans['data'] = [incorrect_evs, incorrect_evcts]
                    answers.append(dict(ans))
        
        incorrect_evcts = linear_algebra.flatten_list(es.eigen_vectors)
        incorrect_evcts = sympy.matrix2numpy(sympy.Matrix(incorrect_evcts).transpose()).tolist()
        incorrect_evs = correct_evs
        if matA != sympy.Matrix(incorrect_evcts):
            ans['feedback'] = r'直交行列で対角化してください。'
            ans['data'] = [incorrect_evs, incorrect_evcts]
            answers.append(dict(ans))
            for i in range(len(correct_evs)):
                incorrect_evs = correct_evs[i:] + correct_evs[:i]
                incorrect_evcts = linear_algebra.flatten_list(es.eigen_vectors)
                if matA != sympy.Matrix(incorrect_evcts):
                    ans['feedback'] = r'直交行列で対角化してください。'
                    ans['data'] = [incorrect_evs, incorrect_evcts]
                    answers.append(dict(ans))
        
        incorrect_evcts = sympy.matrix2numpy(sympy.Matrix(correct_evcts).transpose()).tolist()
        incorrect_evs = correct_evs
        if not self._is_correct(matA, incorrect_evs, incorrect_evcts):
            ans['feedback'] = '対角化に用いる正則な行列の列ベクトルが，固有ベクトルに対応します。'
            ans['data'] = [incorrect_evs, incorrect_evcts]
            answers.append(dict(ans))
        for i in range(len(correct_evs)):
            incorrect_evs = correct_evs[i:] + correct_evs[:i]
            incorrect_evcts = correct_evcts
            if not self._is_correct(matA, incorrect_evs, incorrect_evcts):
                ans['feedback'] = '対角化に用いる正則な行列の列ベクトルは，位置関係から対応する固有値の固有空間の固有ベクトルに対応します。'
                ans['data'] = [incorrect_evs, incorrect_evcts]
                answers.append(dict(ans))
            incorrect_evcts = sympy.matrix2numpy(sympy.Matrix(incorrect_evcts).transpose()).tolist()
            if not self._is_correct(matA, incorrect_evs, incorrect_evcts):
                ans['feedback'] = '対角化に用いる正則な行列の列ベクトルは，位置関係から対応する固有値の固有空間の固有ベクトルに対応します。'
                ans['data'] = [incorrect_evs, incorrect_evcts]
                answers.append(dict(ans))
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        es = quiz.data[0]
        _text = r'次の線形変換（表現行列が実対称行列）について，その固有空間を参考に，その表現表列\(A\)（標準基底に関する）を直交行列により対角化してください。<br />'
        _text += es.str_map(is_latex_closure=True) + r'<br />'
        _text += es.str_eigen_spaces(is_latex_closure=True)
        return _text
    def answer_text(self, ans):
        evs = ans['data'][0]
        evcts = ans['data'][1]
        _matP = sympy.Matrix(evcts).transpose()
        _matD = sympy.diag(*evs)
        _text = r'\(P=' + sympy.latex(_matP, mat_delim='', mat_str='pmatrix') + r',\;'
        _text += r'P^{-1}AP=' + sympy.latex(_matD, mat_delim='', mat_str='pmatrix') + r'\)'
        return _text


# In[35]:


if __name__ == "__main__":
    q = diagonalization_of_real_symmetric_matrix()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[36]:


if __name__ == "__main__":
    pass
    #qz.save('diagonalization_of_real_symmetric_matrix.xml')


# # All the questions

# In[ ]:


questions_str = ['eigenvalue_for_eigenvector', 'select_eigenvector', 'characteristic_polynomial', 
                 'eigen_space_for_numeric_vector_space', 'eigen_space_for_polynomial_vector_space', 
                 'select_diagonalizable_representation_matrix', 'diagonalization_from_numeric_eigen_spaces',
                 'diagonalization_from_polynomial_eigen_spaces', 'diagonalization_of_matrix', 
                 'diagonalization_of_map', 'standard_inner_product_with_norm_in_Rn', 
                 'another_inner_product_with_norm_in_Rn', 'some_inner_product_with_norm_in_Rnm', 
                 'expanding_inner_expression_with_numerical_values', 
                 'expanding_inner_expression_without_numerical_values', 'basis_of_orthogonal_complement', 
                 'orthonomal_basis_in_Rn_with_standard_inner_product', 
                 'orthogonalization_process_in_Rn_with_standard_inner_product', 
                 'normalization_process_in_Rn_with_standard_inner_product', 
                 'orthogonalization_process_in_Rn_with_some_inner_product', 
                 'normalization_process_in_Rn_with_some_inner_product', 'some_inner_product_with_norm_in_Rx', 
                 'orthogonalization_process_in_Rx_with_some_inner_product', 
                 'normalization_process_in_Rx_with_some_inner_product', 'select_orthogonal_matrix', 
                 'unknown_element_in_orthogonal_matrix', 'select_symmetric_matrix', 'select_unitary_matrix', 
                 'select_Hermitian_matrix', 'select_normal_matrix', 'diagonalization_of_real_symmetric_matrix']
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




