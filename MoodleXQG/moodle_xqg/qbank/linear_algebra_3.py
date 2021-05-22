#!/usr/bin/env python
# coding: utf-8

# Moodle XML Question Generator, (C) 2019 Kosaku Nagasaka (Kobe University)

# ## モジュールファイルの生成

# In[1]:


if __name__ == "__main__":
    import subprocess
    subprocess.run(['jupyter', 'nbconvert', '--to', 'python', '_linear_algebra_3.ipynb','--output','linear_algebra_3.py'])


# # Linear Algebra 3 (subspace and linear mapping)

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

# ## general integer set operations

# In[3]:


class general_set_operations(core.Question):
    name = '集合の基本的な操作'
    def __init__(self, emin=1, emax=5, dmin=1, dmax=3):
        # 生成する個々の集合の要素数の範囲
        self.emin = emin
        self.emax = emax
        # 生成する演算の深さの範囲
        self.dmin = dmin
        self.dmax = dmax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='集合の基本的な操作', quiz_number=_quiz_number)
        universe = range(2*(self.emax-self.emin+1))
        depth = random.randint(self.dmin, self.dmax)
        set_expr = linear_algebra.generate_integer_set_expr(depth, universe, self.emin, self.emax)
        set_eval = linear_algebra.eval_integer_set_expr(set_expr)
        set_text = linear_algebra.integer_set_expr_to_text(set_expr)
        quiz.data = [set_expr, set_eval, set_text, universe]
        quiz.quiz_identifier = hash(set_text)
        # 正答の選択肢の生成
        ans = { 'fraction': 100, 'data': set_eval }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        correct_expr_str = str(quiz.data[0])
        # union <=> intersection
        incorrect_expr_str = correct_expr_str.replace('intersection', 'intersectionOLD')
        incorrect_expr_str = incorrect_expr_str.replace('union', 'intersection')
        incorrect_expr_str = incorrect_expr_str.replace('intersectionOLD', 'union')
        incorrect_eval = linear_algebra.eval_integer_set_expr(eval(incorrect_expr_str))
        if incorrect_eval != quiz.data[1]:
            ans['data'] = incorrect_eval
            ans['feedback'] = '和集合と共通部分を取り違えています。'
            answers.append(dict(ans))
        # intersection <=> difference
        incorrect_expr_str = correct_expr_str.replace('intersection', 'intersectionOLD')
        incorrect_expr_str = incorrect_expr_str.replace('difference', 'intersection')
        incorrect_expr_str = incorrect_expr_str.replace('intersectionOLD', 'difference')
        incorrect_eval = linear_algebra.eval_integer_set_expr(eval(incorrect_expr_str))
        if incorrect_eval != quiz.data[1]:
            ans['data'] = incorrect_eval
            ans['feedback'] = '共通部分と差集合を取り違えています。'
            answers.append(dict(ans))
        # union <=> difference
        incorrect_expr_str = correct_expr_str.replace('union', 'unionOLD')
        incorrect_expr_str = incorrect_expr_str.replace('difference', 'union')
        incorrect_expr_str = incorrect_expr_str.replace('unionOLD', 'difference')
        incorrect_eval = linear_algebra.eval_integer_set_expr(eval(incorrect_expr_str))
        if incorrect_eval != quiz.data[1]:
            ans['data'] = incorrect_eval
            ans['feedback'] = '和集合と差集合を取り違えています。'
            answers.append(dict(ans))
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        # ランダムに誤答を作る（本来は推奨されない）
        count = 0
        len_1 = min(len(quiz.data[1]) + 1, len(quiz.data[3]))
        while len(answers) < size and count < 10:
            count += 1
            ans['data'] = set(random.sample(quiz.data[3], len_1))
            ans['feedback'] = '丁寧に集合の計算をし直してみましょう。'
            if ans['data'] != quiz.data[1]:
                answers.append(dict(ans))
            answers = common.answer_union(answers)
        return answers    
    def question_text(self, quiz):
        return '次の集合と一致するものを選んでください。なお，\\( A \\setminus B \\) は，AからBに含まれる要素を取り除く操作を表します。<br />' + quiz.data[2]
    def answer_text(self, ans):
        return common.sympy_expr_to_text(ans['data'])


# In[4]:


if __name__ == "__main__":
    q = general_set_operations()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=2)


# In[5]:


if __name__ == "__main__":
    pass
    #qz.save('general_set_operations.xml')


# ## general set notations

# In[6]:


class general_set_notations(core.Question):
    name = '集合の定義の理解'
    def __init__(self, emin=-5, emax=5, dmin=1, dmax=3, rdefimin=1, rdefimax=2, rdunmin=1, rdunmax=2):
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
        quiz = core.Quiz(name='集合の定義の理解', quiz_number=_quiz_number)
        is_integer = True if random.random() < 0.5 else False
        is_generator = True if random.random() < 0.5 else False
        ls = linear_algebra.LinearSpace()
        ls.set_dimension_range(self.dmin, self.dmax)
        ls.set_element_range(self.emin, self.emax)
        ls.set_rank_deficient_range(self.rdefimin, self.rdefimax)
        ls.set_redundant_range(self.rdunmin, self.rdunmax)
        ls.set_keep_integer(is_integer)
        ls.generate()
        if is_generator:
            vec = [[ele] for ele in ls.a_spanned_vector(is_integer_coefs=is_integer)]
        else:
            vec = [[ele] for ele in ls.a_solution_vector(is_integer_coefs=is_integer)]            
        quiz.data = [is_integer, is_generator, ls]
        quiz.quiz_identifier = hash(ls)
        # 正答の選択肢の生成
        ans = { 'fraction': 100, 'data': vec }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        is_integer = quiz.data[0]
        is_generator = quiz.data[1]
        ls = quiz.data[2]
        # different dimension
        dim = ls.dimension + 1
        ans['data'] = [[linear_algebra.nonzero_randint(self.emin, self.emax)] for i in range(dim)]
        ans['feedback'] = 'ベクトルの次元を確認しましょう。'
        answers.append(dict(ans))
        if ls.dimension > 1:
            dim = ls.dimension - 1
            ans['data'] = [[linear_algebra.nonzero_randint(self.emin, self.emax)] for i in range(dim)]
            ans['feedback'] = 'ベクトルの次元を確認しましょう。'
            answers.append(dict(ans))
        # transpose      
        if ls.dimension > 1:
            if is_generator:
                ans['data'] = [ls.a_spanned_vector(is_integer_coefs=is_integer)]
            else:
                ans['data'] = [ls.a_solution_vector(is_integer_coefs=is_integer)]
            ans['feedback'] = 'ベクトルの向きを確認しましょう。'
            answers.append(dict(ans))        
            dim = ls.dimension + 1
            ans['data'] = [[linear_algebra.nonzero_randint(self.emin, self.emax) for i in range(dim)]]
            ans['feedback'] = 'ベクトルの向きを確認しましょう。'
            answers.append(dict(ans))
            dim = ls.dimension - 1
            ans['data'] = [[linear_algebra.nonzero_randint(self.emin, self.emax) for i in range(dim)]]
            ans['feedback'] = 'ベクトルの向きを確認しましょう。'
            answers.append(dict(ans))
        # domain mismatch
        ans['feedback'] = 'ベクトルの要素が含まれる集合を確認しましょう。'
        if is_integer:
            if is_generator:
                ans['data'] = ls.a_spanned_vector(is_integer_coefs=False)
                if not linear_algebra.is_integer_vector(ans['data']):
                    ans['data'] = [[ele] for ele in ans['data']]
                    answers.append(dict(ans))
            else:
                ans['data'] = ls.a_solution_vector(is_integer_coefs=False)
                if not linear_algebra.is_integer_vector(ans['data']):
                    ans['data'] = [[ele] for ele in ans['data']]
                    answers.append(dict(ans))
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        # ランダムに誤答を作る（本来は推奨されない）
        count = 0
        while len(answers) < size and count < 10:
            count += 1
            ans['feedback'] = '丁寧に集合に含まれるかを確認し直しましょう。'
            if is_generator:
                _vec = [linear_algebra.nonzero_randint(self.emin, self.emax) for i in range(ls.dimension)]
                if not ls.is_a_spanned_vector(_vec):
                    ans['data'] = [[ele] for ele in _vec]
                    answers.append(dict(ans))
            else:
                _vec = [linear_algebra.nonzero_randint(self.emin, self.emax) for i in range(ls.dimension)]
                if not ls.is_a_solution_vector(_vec):
                    ans['data'] = [[ele] for ele in _vec]
                    answers.append(dict(ans))
            answers = common.answer_union(answers)
        return answers    
    def question_text(self, quiz):
        _text = '次の集合に含まれるベクトルを選んでください。<br />'
        if quiz.data[0]:
            _domain = 'Z'
        else:
            _domain = 'R'
        if quiz.data[1]:
            _text += quiz.data[2].str_as_spanned_space(domain=_domain)
        else:
            _text += quiz.data[2].str_as_solution_space(domain=_domain,is_homogeneous=False)
        return  _text
    def answer_text(self, ans):
        return common.sympy_expr_to_text(sympy.Matrix(ans['data']))


# In[7]:


if __name__ == "__main__":
    q = general_set_notations()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=2)


# In[8]:


if __name__ == "__main__":
    pass
    #qz.save('general_set_notations.xml')


# ## same sub-space recognition

# In[39]:


class same_subspace_recognition(core.Question):
    name = '同一の部分空間の認識'
    def __init__(self, emin=-3, emax=3, dmin=2, dmax=3, rdefimin=1, rdefimax=1, rdunmin=1, rdunmax=2):
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
        quiz = core.Quiz(name='同一の部分空間の認識', quiz_number=_quiz_number)
        is_polynomial = True if random.random() < 0.35 else False
        is_generator = True if is_polynomial or random.random() < 0.5 else False        
        ls = linear_algebra.LinearSpace()
        ls.set_dimension_range(self.dmin, self.dmax)
        ls.set_element_range(self.emin, self.emax)
        ls.set_rank_deficient_range(self.rdefimin, self.rdefimax)
        ls.set_redundant_range(self.rdunmin, self.rdunmax)
        ls.set_keep_integer(False)
        ls.generate()
        quiz.data = [is_polynomial, is_generator, ls]
        quiz.quiz_identifier = hash(ls)
        # 正答の選択肢の生成
        ls_sub = linear_algebra.LinearSpace(ls)
        if is_polynomial:
            ls_sub.generate_generator()
            _alternative = ls_sub.str_as_spanned_space(is_polynomial=True)
        elif is_generator:
            if random.random() < 0.5:
                ls_sub.generate_generator()
                _alternative = ls_sub.str_as_spanned_space(is_polynomial=False)
            else:
                _alternative = ls_sub.str_as_solution_space()
        else:
            _alternative = ls_sub.str_as_spanned_space(is_polynomial=False)        
        ans = { 'fraction': 100, 'data': _alternative }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        is_polynomial = quiz.data[0]
        is_generator = quiz.data[1]
        ls = quiz.data[2]
        # different domain
        if is_polynomial:
            ans['data'] = ls.str_as_spanned_space(is_polynomial=False)
        elif is_generator:
            ans['data'] = ls.str_as_spanned_space(is_polynomial=True)
        else:
            ans['data'] = ls.str_as_spanned_space(domain='Z')
        ans['feedback'] = 'ベクトルの要素が含まれる集合を確認しましょう。'
        answers.append(dict(ans))
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        # ランダムに誤答を作る（本来は推奨されない）
        count = 0
        while len(answers) < size and count < 10:
            count += 1
            ls_sub = linear_algebra.LinearSpace(ls)
            ls_sub.generate_conditions()
            ls_sub.generate()
            if ls_sub != ls:
                if is_polynomial:
                    if random.random() < 0.333 and ls_sub.num_disj != ls.num_disj:
                        ans['data'] = ls_sub.str_as_spanned_space(is_polynomial=True)
                        ans['feedback'] = 'ベクトルの次元を確認しましょう。'
                    elif random.random() < 0.5:
                        ans['data'] = ls_sub.str_as_spanned_space(is_polynomial=False)
                        ans['feedback'] = 'ベクトルの要素が含まれる集合を確認しましょう。'
                    else:
                        ans['data'] = ls_sub.str_as_solution_space()
                        ans['feedback'] = 'ベクトルの要素が含まれる集合を確認しましょう。'
                elif is_generator:
                    if random.random() < 0.333 and ls_sub.dimension != ls.dimension:
                        ans['data'] = ls_sub.str_as_spanned_space(is_polynomial=False)
                        ans['feedback'] = 'ベクトルの次元を確認しましょう。'
                    elif random.random() < 0.5:
                        ans['data'] = ls_sub.str_as_spanned_space(is_polynomial=True)
                        ans['feedback'] = 'ベクトルの要素が含まれる集合を確認しましょう。'
                    elif ls_sub.dimension != ls.dimension:
                        ans['data'] = ls_sub.str_as_solution_space()
                        ans['feedback'] = 'ベクトルの次元を確認しましょう。'
                else:
                    if random.random() < 0.333 and ls_sub.dimension != ls.dimension:
                        ans['data'] = ls_sub.str_as_spanned_space(is_polynomial=False)
                        ans['feedback'] = 'ベクトルの次元を確認しましょう。'
                    elif random.random() < 0.5:
                        ans['data'] = ls_sub.str_as_spanned_space(is_polynomial=True)
                        ans['feedback'] = 'ベクトルの要素が含まれる集合を確認しましょう。'
                    elif ls_sub.dimension != ls.dimension:
                        ans['data'] = ls_sub.str_as_solution_space()
                        ans['feedback'] = 'ベクトルの次元を確認しましょう。'
                answers.append(dict(ans))
                answers = common.answer_union(answers) 
        return answers    
    def question_text(self, quiz):
        _text = '次の部分空間と同じ部分空間を選択してください。<br />'
        if quiz.data[0]:
            _text += quiz.data[2].str_as_spanned_space(is_polynomial=True)
        elif quiz.data[1]:
            _text += quiz.data[2].str_as_spanned_space(is_polynomial=False)
        else:
            _text += quiz.data[2].str_as_solution_space()
        return  _text
    def answer_text(self, ans):
        return ans['data']


# In[40]:


if __name__ == "__main__":
    q = same_subspace_recognition()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=2)


# In[41]:


if __name__ == "__main__":
    pass
    #qz.save('same_subspace_recognition.xml')


# ## which is a sub space

# In[12]:


class a_subspace_recognition(core.Question):
    name = '部分空間の選択'
    def __init__(self, emin=-3, emax=3, dmin=2, dmax=3):
        # 生成する個々の数の範囲
        self.emin = emin
        self.emax = emax
        # 生成するベクトルの次元の範囲
        self.dmin = dmin
        self.dmax = dmax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='部分空間の選択', quiz_number=_quiz_number)
        ls = linear_algebra.LinearSubSpace()
        ls.set_dimension_range(self.dmin, self.dmax)
        ls.set_element_range(self.emin, self.emax)
        ls.generate(is_subspace=True)
        quiz.data = None
        quiz.quiz_identifier = hash(ls)
        # 正答の選択肢の生成
        ans = { 'fraction': 100, 'data': ls.str() }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        ans['feedback'] = '部分空間の必要条件（零ベクトルの存在）や必要十分条件を確認しましょう。'
        ls = linear_algebra.LinearSubSpace()
        ls.set_dimension_range(self.dmin, self.dmax)
        ls.set_element_range(self.emin, self.emax)
        count = 0
        while len(answers) < size and count < 10:
            count += 1
            ls.generate(is_subspace=False)
            ans['data'] = ls.str()
            answers.append(dict(ans))
            answers = common.answer_union(answers) 
        return answers    
    def question_text(self, quiz):
        _text = '次の中から' + r'\( \mathbb{R} \)' + '上のベクトル空間の部分空間になっているものを選択してください。'
        return _text
    def answer_text(self, ans):
        return ans['data']


# In[13]:


if __name__ == "__main__":
    q = a_subspace_recognition()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=2)


# In[14]:


if __name__ == "__main__":
    pass
    #qz.save('a_subspace_recognition.xml')


# ## representable vector by linear combination

# In[15]:


class representable_vector_by_linear_combination(core.Question):
    name = '線形結合で表現可能なベクトル'
    def __init__(self, emin=-3, emax=3, dmin=3, dmax=4, rdefimin=1, rdefimax=1, rdunmin=1, rdunmax=1):
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
        quiz = core.Quiz(name='線形結合のベクトル', quiz_number=_quiz_number)
        ls = linear_algebra.LinearSpace()
        ls.set_dimension_range(self.dmin, self.dmax)
        ls.set_element_range(self.emin, self.emax)
        ls.set_rank_deficient_range(self.rdefimin, self.rdefimax)
        ls.set_redundant_range(self.rdunmin, self.rdunmax)
        ls.set_keep_integer(True)
        ls.generate()
        vec = ls.a_spanned_vector(is_integer_coefs=True)
        _is_matrix = False
        _is_polynomial = False
        if random.random() < 0.334:
            _is_matrix = True
        elif random.random() < 0.5:
            _is_polynomial = True
        quiz.data = [ls, _is_polynomial, _is_matrix, vec]
        quiz.quiz_identifier = hash(ls)
        # 正答の選択肢の生成
        ans = { 'fraction': 100, 'data': ls.a_vector_as_str(vec, is_polynomial=_is_polynomial, is_matrix=_is_matrix) }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        ls = quiz.data[0]
        _is_polynomial = quiz.data[1]
        _is_matrix = quiz.data[2]
        vec = quiz.data[3]
        # different dimension
        ans['feedback'] = 'ベクトルの次元を確認しましょう。'
        if not _is_matrix:
            dim = ls.dimension + 1
            ans['data'] = ls.a_vector_as_str([linear_algebra.nonzero_randint(self.emin, self.emax) for i in range(dim)], 
                                             is_polynomial=_is_polynomial, is_matrix=_is_matrix)
            answers.append(dict(ans))
            if ls.dimension > 1:
                dim = ls.dimension - 1
                _vec = [linear_algebra.nonzero_randint(self.emin, self.emax) for i in range(dim)]
                if not ls.is_a_spanned_vector(_vec):
                    ans['data'] = ls.a_vector_as_str(_vec, is_polynomial=_is_polynomial, is_matrix=_is_matrix)
                    answers.append(dict(ans))
        # different space
        ans['feedback'] = 'ベクトルの種類を確認しましょう。'
        if _is_matrix:
            ans['data'] = ls.a_vector_as_str(vec, is_polynomial=False, is_matrix=False)
            answers.append(dict(ans))
            ans['data'] = ls.a_vector_as_str(vec, is_polynomial=True, is_matrix=False)
            answers.append(dict(ans))
        elif _is_polynomial:
            ans['data'] = ls.a_vector_as_str(vec, is_polynomial=False, is_matrix=False)
            answers.append(dict(ans))
            ans['data'] = ls.a_vector_as_str(vec, is_polynomial=False, is_matrix=True)
            answers.append(dict(ans))
        else:
            ans['data'] = ls.a_vector_as_str(vec, is_polynomial=True, is_matrix=False)
            answers.append(dict(ans))
            ans['data'] = ls.a_vector_as_str(vec, is_polynomial=False, is_matrix=True)
            answers.append(dict(ans))
        # ランダムに誤答を作る（本来は推奨されない）
        count = 0
        while len(answers) < size*3 and count < 100:
            count += 1
            _vec = [linear_algebra.nonzero_randint(self.emin, self.emax) for i in range(ls.dimension)]
            if not ls.is_a_spanned_vector(_vec):
                if random.random() < 0.334:
                    ans['data'] = ls.a_vector_as_str(_vec, is_polynomial=False, is_matrix=False)
                    if (not _is_polynomial) and (not _is_matrix):
                        ans['feedback'] = '丁寧に線形結合可能であるかを確認し直しましょう。'
                    else:
                        ans['feedback'] = 'ベクトルの種類を確認しましょう。'
                elif random.random() < 0.5:
                    ans['data'] = ls.a_vector_as_str(_vec, is_polynomial=True, is_matrix=False)
                    if _is_polynomial and (not _is_matrix):
                        ans['feedback'] = '丁寧に線形結合可能であるかを確認し直しましょう。'
                    else:
                        ans['feedback'] = 'ベクトルの種類を確認しましょう。'
                else:                    
                    ans['data'] = ls.a_vector_as_str(_vec, is_polynomial=False, is_matrix=True)
                    if (not _is_polynomial) and _is_matrix:
                        ans['feedback'] = '丁寧に線形結合可能であるかを確認し直しましょう。'
                    else:
                        ans['feedback'] = 'ベクトルの種類を確認しましょう。'
                answers.append(dict(ans))
            answers = common.answer_union(answers)
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        _text = '次のベクトルの一次結合（線形結合）で表せるベクトルを選択してください。<br />' + r'\( '
        ls = quiz.data[0]
        _is_polynomial = quiz.data[1]
        _is_matrix = quiz.data[2]
        generator = ls.generator
        for i in range(len(generator)):
            _text += ls.a_vector_as_str(generator[i], is_latex_closure=False, is_polynomial=_is_polynomial, is_matrix=_is_matrix)
            if i < len(generator) - 1:
                _text += r',\;'        
        return  _text + r' \)'
    def answer_text(self, ans):
        return ans['data']


# In[16]:


if __name__ == "__main__":
    q = representable_vector_by_linear_combination()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=2)


# In[17]:


if __name__ == "__main__":
    pass
    #qz.save('representable_vector_by_linear_combination.xml')


# ## non-trivial linear relation vector

# In[18]:


class nontrivial_linear_relation_vector(core.Question):
    name = '非自明な線形関係を持つベクトルの組を選ぶ'
    def __init__(self, emin=-3, emax=3, dmin=2, dmax=4, rdefimin=0, rdefimax=2):
        # 生成する個々の数の範囲
        self.emin = emin
        self.emax = emax
        # 生成するベクトルの次元の範囲
        self.dmin = dmin
        self.dmax = dmax
        # 生成するベクトルの次元からのランク落ちの範囲
        self.rdefimin = rdefimin
        self.rdefimax = rdefimax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='非自明線形関係のベクトル', quiz_number=_quiz_number)
        ls = linear_algebra.LinearSpace()
        ls.set_dimension_range(self.dmin, self.dmax)
        ls.set_element_range(self.emin, self.emax)
        ls.set_rank_deficient_range(self.rdefimin, self.rdefimax)
        ls.set_redundant_range(1, 1) # make generators redundant
        ls.set_keep_integer(True)
        ls.generate()
        quiz.data = None
        quiz.quiz_identifier = hash(ls)
        # 正答の選択肢の生成
        _is_polynomial = False
        _is_matrix = False
        if ls.dimension == 4:
            _is_matrix = True
        elif random.random() < 0.5:
            _is_polynomial = True
        ans = { 'fraction': 100, 'data': self._str_list_vectors(ls, _is_polynomial, _is_matrix) }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        ls = linear_algebra.LinearSpace()
        ls.set_dimension_range(self.dmin, self.dmax)
        ls.set_element_range(self.emin, self.emax)
        ls.set_rank_deficient_range(self.rdefimin, self.rdefimax)
        ls.set_redundant_range(0, 0) # make generators not-redundant
        ls.set_keep_integer(True)
        # ランダムに誤答を作る（本来は推奨されない）
        count = 0
        while len(answers) < size and count < 100:
            count += 1
            ls.generate()
            while ls.num_gens == 1:
                ls.generate()
            _is_polynomial = False
            _is_matrix = False
            if ls.dimension == 4:
                _is_matrix = True
                ans['feedback'] = 'ベクトル空間として見る場合，行列は1列ないしは1行に並び直して，普通のベクトルとして考えても同じです。'
            elif random.random() < 0.5:
                _is_polynomial = True
                ans['feedback'] = 'ベクトル空間として見る場合，多項式は係数のみを取り出して，普通のベクトルとして考えても同じです。'
            else:
                ans['feedback'] = '線形関係に関する同次線形方程式が非自明解を持つか確認しましょう。'
            ans['data'] = self._str_list_vectors(ls, _is_polynomial, _is_matrix)
            answers.append(dict(ans))
            answers = common.answer_union(answers)
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        _text = '次のベクトルの組のうち，非自明な線形関係を持つ組（零ベクトルを線形結合可能である組）を選択してください。'
        return _text    
    def answer_text(self, ans):
        return ans['data']
    def _str_list_vectors(self, ls, _is_polynomial, _is_matrix):
        _text = r'\( '
        for i in range(ls.num_gens):
            _text += ls.a_vector_as_str(ls.generator[i], is_latex_closure=False, is_polynomial=_is_polynomial, is_matrix=_is_matrix)
            if i < ls.num_gens - 1:
                _text += r',\;'        
        return  _text + r' \)'


# In[19]:


if __name__ == "__main__":
    q = nontrivial_linear_relation_vector()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=2)


# In[20]:


if __name__ == "__main__":
    pass
    #qz.save('nontrivial_linear_relation_vector.xml')


# ## linearly independent check equation

# In[21]:


class linearly_independent_check_equation(core.Question):
    name = '線形独立性の確認に必要な同次線形方程式'
    def __init__(self, emin=-3, emax=3, dmin=2, dmax=4, rdefimin=0, rdefimax=1, rdunmin=0, rdunmax=1):
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
        quiz = core.Quiz(name='線形独立確認の方程式', quiz_number=_quiz_number)
        ls = linear_algebra.LinearSpace()
        ls.set_dimension_range(self.dmin, self.dmax)
        ls.set_element_range(self.emin, self.emax)
        ls.set_rank_deficient_range(self.rdefimin, self.rdefimax)
        ls.set_redundant_range(self.rdunmin, self.rdunmax)
        ls.set_keep_integer(True)
        ls.generate()
        while ls.num_gens < 2:
            ls.generate()
        _is_polynomial = False
        _is_matrix = False
        if ls.dimension == 4 and random.random() < 0.5:
            _is_matrix = True
        elif random.random() < 0.5:
            _is_polynomial = True
        quiz.data = [ls, _is_polynomial, _is_matrix]
        quiz.quiz_identifier = hash(ls)
        # 正答の選択肢の生成
        ans = { 'fraction': 100, 'data': self._str_linear_eq(sympy.Matrix(random.sample(ls.generator, len(ls.generator))).transpose()) }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        ls = quiz.data[0]
        # ランダムに誤答を作る（本来は推奨されない）
        count = 0
        while len(answers) < size and count < 100:
            count += 1
            ls_sub = linear_algebra.LinearSpace(ls)
            _matrix = random.sample(ls_sub.generator, len(ls_sub.generator))
            r = random.randint(0, ls_sub.num_gens - 1)
            c = random.randint(0, ls_sub.dimension - 1)
            _matrix[r][c] += random.choice([-1,1])
            if random.random() < 0.5:
                ans['feedback'] = '未知数は，それぞれのベクトルに掛け合わされることに注意して読み解きましょう。'
                ans['data'] = self._str_linear_eq(sympy.Matrix(_matrix)) 
            else:
                ans['feedback'] = 'よく確認しましょう。線形独立性を確認する場合，ベクトルの順序は関係ありません。'
                ans['data'] = self._str_linear_eq(sympy.Matrix(_matrix).transpose()) 
            answers.append(dict(ans))
            answers = common.answer_union(answers)
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        _text = '次のベクトルが線形独立であるかを定義に基づき確認する場合に構成すべき同次線形方程式を選択してください。<br />'
        _text += self._str_list_vectors(quiz.data[0], quiz.data[1], quiz.data[2])
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
    def _str_list_vectors(self, ls, _is_polynomial, _is_matrix):
        _text = r'\( '
        for i in range(ls.num_gens):
            _text += ls.a_vector_as_str(ls.generator[i], is_latex_closure=False, is_polynomial=_is_polynomial, is_matrix=_is_matrix)
            if i < ls.num_gens - 1:
                _text += r',\;'        
        return _text + r' \)'


# In[22]:


if __name__ == "__main__":
    q = linearly_independent_check_equation()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=2)


# In[23]:


if __name__ == "__main__":
    pass
    #qz.save('linearly_independent_check_equation.xml')


# ## linearly independent check equation with basis

# In[43]:


class linearly_independent_check_equation_with_basis(core.Question):
    name = '線形独立性の確認に必要な同次線形方程式（他のベクトルの線形結合で与えられた場合）'
    def __init__(self, emin=-3, emax=3, dmin=2, dmax=3, rdefimin=0, rdefimax=1, rdunmin=0, rdunmax=1):
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
        quiz = core.Quiz(name='線形独立確認の方程式（独立ベクトルあり）', quiz_number=_quiz_number)
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
        # 正答の選択肢の生成
        ans = { 'fraction': 100, 'data': self._str_linear_eq(sympy.Matrix(random.sample(ls.generator, len(ls.generator))).transpose()) }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
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
        # ランダムに誤答を作る（本来は推奨されない）
        count = 0
        while len(answers) < size and count < 100:
            count += 1
            ls_sub = linear_algebra.LinearSpace(ls)
            _matrix = random.sample(ls_sub.generator, len(ls_sub.generator))
            r = random.randint(0, ls_sub.num_gens - 1)
            c = random.randint(0, ls_sub.dimension - 1)
            _matrix[r][c] += random.choice([-1,1])
            if random.random() < 0.5:
                ans['feedback'] = '未知数は，それぞれのベクトルに掛け合わされることに注意して読み解きましょう。'
                ans['data'] = self._str_linear_eq(sympy.Matrix(_matrix)) 
            else:
                ans['feedback'] = 'よく確認しましょう。線形独立性を確認する場合，ベクトルの順序は関係ありません。'
                ans['data'] = self._str_linear_eq(sympy.Matrix(_matrix).transpose()) 
            answers.append(dict(ans))
            answers = common.answer_union(answers)
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        _text = 'ベクトル' + self._str_us(quiz.data) + 'が線形独立であるとき，次のベクトル' + self._str_vs(quiz.data)
        _text += 'が線形独立であるか定義に基づき確認する場合に構成すべき同次線形方程式を選択してください。<br />'
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


# In[44]:


if __name__ == "__main__":
    q = linearly_independent_check_equation_with_basis()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=2)


# In[45]:


if __name__ == "__main__":
    pass
    #qz.save('linearly_independent_check_equation_with_basis.xml')


# ## reduced row echelon form and linearly dependent relation

# In[31]:


class rref_and_linearly_dependent_relation(core.Question):
    name = '掃き出し法の結果と線形従属なベクトルとの関係'
    def __init__(self, emin=-3, emax=3, dmin=3, dmax=5, rdefimin=1, rdefimax=3, rdunmin=0, rdunmax=0):
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
        quiz = core.Quiz(name='掃き出し結果と従属性', quiz_number=_quiz_number)
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
        # 正答の選択肢の生成
        ans = { 'fraction': 100, 'data': self._str_rref_to_us(ls, conv_pivot=ls.pivot_positions)}
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
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
        ans['feedback'] = '行の基本変形では，列間の関係性は変化しません。符号に注意して確認しましょう。'
        ans['data'] = self._str_rref_to_us(ls_sub, conv_pivot=ls_sub.pivot_positions)
        answers.append(dict(ans))
        # gather pivots
        if ls.pivot_positions[-1] != len(ls.pivot_positions)-1 and ls.pivot_positions[-1] < ls.dimension - 1:
            _conv = [i for i in range(len(ls_sub.pivot_positions))]
            ans['feedback'] = '主成分の位置と意味に注意してください。主成分のある列ベクトルを表しています。また，符号にも注意して確認しましょう。'
            ans['data'] = self._str_rref_to_us(ls_sub, conv_pivot=_conv)
            answers.append(dict(ans))
            ls_sub = linear_algebra.LinearSpace(ls)
            ans['feedback'] = '主成分の位置と意味に注意してください。主成分のある列ベクトルを表しています。'
            ans['data'] = self._str_rref_to_us(ls_sub, conv_pivot=_conv)
            answers.append(dict(ans))
        # ランダムに誤答を作る（本来は推奨されない）
        count = 0
        while len(answers) < size and count < 100:
            count += 1
            ls_sub = linear_algebra.LinearSpace(ls)
            ls_sub.generate_basis()
            if ls_sub.basis != ls.basis:
                ans['feedback'] = '行の基本変形では，列間の関係性は変化しません。主成分の位置と意味に注意してください。'
                ans['data'] = self._str_rref_to_us(ls_sub, conv_pivot=ls_sub.pivot_positions)
                answers.append(dict(ans))
                answers = common.answer_union(answers)
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        _text = '次のベクトル' + self._str_us(quiz.data) + 'の線形独立な最大個数を調べるため，'
        _text += r'行列' + self._str_us_matrix(quiz.data) + 'に掃き出しを行ったところ，'
        _text += r'次の行列\( A \)が得られた。この状態で，主成分のある列を線形独立なベクトルとして取り出し，'
        _text += '残りのベクトルをその線形結合で表した場合の関係式を選んでください。<br />'
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


# In[32]:


if __name__ == "__main__":
    q = rref_and_linearly_dependent_relation()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=2)


# In[33]:


if __name__ == "__main__":
    pass
    #qz.save('rref_and_linearly_dependent_relation.xml')


# ## maximum linearly independent set

# In[26]:


class maximum_linearly_independent_set(core.Question):
    name = '線形独立な最大個数とそのベクトルの組'
    def __init__(self, emin=-3, emax=3, dmin=2, dmax=4, rdefimin=0, rdefimax=1, rdunmin=0, rdunmax=2):
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
        # dummy
        self.ls_dummy = linear_algebra.LinearSpace()
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='線形独立な最大個数', quiz_number=_quiz_number)
        ls = linear_algebra.LinearSpace()
        ls.set_dimension_range(self.dmin, self.dmax)
        ls.set_element_range(self.emin, self.emax)
        ls.set_rank_deficient_range(self.rdefimin, self.rdefimax)
        ls.set_redundant_range(self.rdunmin, self.rdunmax)
        ls.set_keep_integer(True)
        ls.generate()
        while ls.num_gens < 2:
            ls.generate()
        _is_polynomial = False
        _is_matrix = False
        quiz.quiz_identifier = hash(ls)
        # 正答の選択肢の生成
        r = ls.num_disj;
        bs = [];
        for ele in itertools.combinations(ls.generator, ls.num_disj):
            if sympy.Matrix(ele).rank() == ls.num_disj:
                bs = ele
                break    
        quiz.data = [ls, r, bs]
        ans = { 'fraction': 100, 'data': [r, bs] }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        ls = quiz.data[0]
        r = quiz.data[1]
        bs = quiz.data[2]
        # 生成系からのランダムと r のずれ
        if ls.num_gens - r != r and random.random() < 0.5:
            ans['feedback'] = '線形独立な最大個数とは，名前の通り，線形独立となるベクトルの組の中で最もベクトルの個数が大きなものを指します。'
            ans['data'] = [ls.num_gens - r, bs]
            answers.append(dict(ans))
        for n in range(1, ls.num_gens + 1):
            for ele in itertools.combinations(ls.generator, n):
                if n == ls.num_disj and sympy.Matrix(ele).rank() == ls.num_disj:
                    continue
                if n < ls.num_disj:
                    ans['feedback'] = 'より多くのベクトルを含む組でも線形独立になるはずです。'
                elif n > ls.num_disj:
                    ans['feedback'] = 'より少ないベクトルを含む組でなければ，線形従属になってしまいます。'
                else:
                    ans['feedback'] = '線形独立となるベクトルの組を探してください。'
                ans['data'] = [n, ele]
                answers.append(dict(ans))
                if ls.num_gens - r != r and random.random() < 0.5:
                    ans['feedback'] = '線形独立な最大個数とは，名前の通り，線形独立となるベクトルの組の中で最もベクトルの個数が大きなものを指します。'
                    ans['data'] = [ls.num_gens - n, ele]
                    answers.append(dict(ans))
                answers = common.answer_union(answers)
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        _text = r'次のベクトルの線形独立な最大個数\( r \)と，\( r \)個の線形独立なベクトルの組を選択してください。<br />'
        _text += self._str_list_vectors(quiz.data[0], False, False)
        return _text
    def answer_text(self, ans):
        _text = r'\( '
        _text += r'r = ' + sympy.latex(ans['data'][0]) + r',\;'
        _text += r'\left\{'
        for i in range(len(ans['data'][1])):
            _text += self.ls_dummy.a_vector_as_str(ans['data'][1][i], is_latex_closure=False, is_polynomial=False, is_matrix=False)
            if i < len(ans['data'][1]) - 1:
                _text += r',\;'         
        _text += r'\right\}'
        return _text + r' \)'
    def _str_list_vectors(self, ls, _is_polynomial, _is_matrix):
        _text = r'\( '
        for i in range(ls.num_gens):
            _text += ls.a_vector_as_str(ls.generator[i], is_latex_closure=False, is_polynomial=_is_polynomial, is_matrix=_is_matrix)
            if i < ls.num_gens - 1:
                _text += r',\;'        
        return _text + r' \)'


# In[27]:


if __name__ == "__main__":
    q = maximum_linearly_independent_set()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=2)


# In[28]:


if __name__ == "__main__":
    pass
    #qz.save('maximum_linearly_independent_set.xml')


# ## maximum linearly independent set general case

# In[37]:


class maximum_linearly_independent_set_general_case(core.Question):
    name = '線形独立な最大個数とそのベクトルの組（一般）'
    def __init__(self, emin=-3, emax=3, dmin=2, dmax=4, rdefimin=0, rdefimax=1, rdunmin=0, rdunmax=2):
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
        # dummy
        self.ls_dummy = linear_algebra.LinearSpace()
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='線形独立な最大個数（一般）', quiz_number=_quiz_number)
        ls = linear_algebra.LinearSpace()
        ls.set_dimension_range(self.dmin, self.dmax)
        ls.set_element_range(self.emin, self.emax)
        ls.set_rank_deficient_range(self.rdefimin, self.rdefimax)
        ls.set_redundant_range(self.rdunmin, self.rdunmax)
        ls.set_keep_integer(True)
        ls.generate()
        while ls.num_gens < 2:
            ls.generate()
        _is_polynomial = False
        _is_matrix = False
        if ls.dimension == 4 and random.random() < 0.5:
            _is_matrix = True
        else:
            _is_polynomial = True
        quiz.quiz_identifier = hash(ls)
        # 正答の選択肢の生成
        r = ls.num_disj;
        bs = [];
        for ele in itertools.combinations(ls.generator, ls.num_disj):
            if sympy.Matrix(ele).rank() == ls.num_disj:
                bs = ele
                break    
        quiz.data = [ls, r, bs, _is_matrix, _is_polynomial]
        ans = { 'fraction': 100, 'data': [r, bs, _is_matrix, _is_polynomial] }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        ls = quiz.data[0]
        r = quiz.data[1]
        bs = quiz.data[2]
        _is_matrix = quiz.data[3]
        _is_polynomial = quiz.data[4]
        # 生成系からのランダムと r のずれ
        if ls.num_gens - r != r and random.random() < 0.5:
            ans['feedback'] = '線形独立な最大個数とは，名前の通り，線形独立となるベクトルの組の中で最もベクトルの個数が大きなものを指します。'
            ans['data'] = [ls.num_gens - r, bs, _is_matrix, _is_polynomial]
            answers.append(dict(ans))
        if ls.num_gens - r != r and random.random() < 0.25:
            ans['feedback'] = '与えられたベクトルから選んでください。'
            ans['data'] = [ls.num_gens - r, bs, False, False]
            answers.append(dict(ans))
        if random.random() < 0.25:
            ans['feedback'] = '与えられたベクトルから選んでください。'
            ans['data'] = [r, bs, False, False]
            answers.append(dict(ans))            
        for n in range(1, ls.num_gens + 1):
            for ele in itertools.combinations(ls.generator, n):
                if n == ls.num_disj and sympy.Matrix(ele).rank() == ls.num_disj:
                    continue
                if n < ls.num_disj:
                    ans['feedback'] = 'より多くのベクトルを含む組でも線形独立になるはずです。'
                elif n > ls.num_disj:
                    ans['feedback'] = 'より少ないベクトルを含む組でなければ，線形従属になってしまいます。'
                else:
                    ans['feedback'] = '線形独立となるベクトルの組を探してください。'
                ans['data'] = [n, ele, _is_matrix, _is_polynomial]
                answers.append(dict(ans))
                if ls.num_gens - r != r and random.random() < 0.5:
                    ans['feedback'] = '線形独立な最大個数とは，名前の通り，線形独立となるベクトルの組の中で最もベクトルの個数が大きなものを指します。'
                    ans['data'] = [ls.num_gens - n, ele, _is_matrix, _is_polynomial]
                    answers.append(dict(ans))
                if ls.num_gens - r != r and random.random() < 0.25:
                    ans['feedback'] = '与えられたベクトルから選んでください。'
                    ans['data'] = [ls.num_gens - n, ele, False, False]
                    answers.append(dict(ans))
                if ls.num_gens - r != r and random.random() < 0.25:
                    ans['feedback'] = '与えられたベクトルから選んでください。'
                    ans['data'] = [n, ele, False, False]
                    answers.append(dict(ans))
                answers = common.answer_union(answers)
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        _text = r'次のベクトルの線形独立な最大個数\( r \)と，\( r \)個の線形独立なベクトルの組を選択してください。<br />'
        _text += self._str_list_vectors(quiz.data[0], quiz.data[4], quiz.data[3])
        return _text
    def answer_text(self, ans):
        _text = r'\( '
        _text += r'r = ' + sympy.latex(ans['data'][0]) + r',\;'
        _text += r'\left\{'
        for i in range(len(ans['data'][1])):
            _text += self.ls_dummy.a_vector_as_str(ans['data'][1][i], is_latex_closure=False, is_polynomial=ans['data'][3], is_matrix=ans['data'][2])
            if i < len(ans['data'][1]) - 1:
                _text += r',\;'         
        _text += r'\right\}'
        return _text + r' \)'
    def _str_list_vectors(self, ls, _is_polynomial, _is_matrix):
        _text = r'\( '
        for i in range(ls.num_gens):
            _text += ls.a_vector_as_str(ls.generator[i], is_latex_closure=False, is_polynomial=_is_polynomial, is_matrix=_is_matrix)
            if i < ls.num_gens - 1:
                _text += r',\;'        
        return _text + r' \)'


# In[38]:


if __name__ == "__main__":
    q = maximum_linearly_independent_set_general_case()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=2)


# In[39]:


if __name__ == "__main__":
    pass
    #qz.save('maximum_linearly_independent_set_general_case.xml')


# ## matrix construction for basis computation

# In[4]:


class matrix_const_for_basis_comp(core.Question):
    name = '基底計算に適する掃き出し対象の行列の構成'
    def __init__(self, emin=-3, emax=3, dmin=2, dmax=4, rdefimin=0, rdefimax=1, rdunmin=0, rdunmax=2):
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
        # dummy
        self.ls_dummy = linear_algebra.LinearSpace()
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='基底計算用の行列構成', quiz_number=_quiz_number)
        ls = linear_algebra.LinearSpace()
        ls.set_dimension_range(self.dmin, self.dmax)
        ls.set_element_range(self.emin, self.emax)
        ls.set_rank_deficient_range(self.rdefimin, self.rdefimax)
        ls.set_redundant_range(self.rdunmin, self.rdunmax)
        ls.set_keep_integer(True)
        ls.generate()
        while ls.num_gens < 2:
            ls.generate()
        _is_polynomial = False
        _is_matrix = False
        if ls.dimension == 4 and random.random() < 0.5:
            _is_matrix = True
        elif random.random() < 0.5:
            _is_polynomial = True
        quiz.quiz_identifier = hash(ls)
        # 正答の選択肢の生成
        quiz.data = [ls, _is_polynomial, _is_matrix]
        ans = { 'fraction': 100, 'data': sympy.matrix2numpy(sympy.Matrix(ls.generator).transpose()).tolist() }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        ls = quiz.data[0]
        _is_polynomial = quiz.data[1]
        _is_matrix = quiz.data[2]
        if sympy.Matrix(ls.generator) != sympy.Matrix(ls.generator).transpose():
            ans['feedback'] = '今回の基底の計算に必要な行列は，列が1つのベクトルに対応する必要があります。'
            ans['data'] = ls.generator
            answers.append(dict(ans))
        if _is_matrix:
            ans['feedback'] = '今回の基底の計算に必要な行列は，列が1つのベクトルに対応する必要があります。'
            ans['data'] = linear_algebra.thread_list([linear_algebra.partition_list(v, 2) for v in ls.generator])
            answers.append(dict(ans))
            ans['feedback'] = '今回の基底の計算に必要な行列は，列が1つのベクトルに対応する必要があります。'
            ans['data'] = linear_algebra.thread_list([sympy.matrix2numpy(sympy.Matrix(linear_algebra.partition_list(v, 2)).transpose()).tolist() for v in ls.generator])
            answers.append(dict(ans))        
        # ランダムに誤答を作る（本来は推奨されない）
        count = 0
        while len(answers) < size*2 and count < 100:
            count += 1
            ls_sub = linear_algebra.LinearSpace(ls)
            r = random.randint(0, ls_sub.num_gens - 1)
            c = random.randint(0, ls_sub.dimension - 1)
            ls_sub.generator[r][c] += random.choice([-2,-1,1,2])
            ans['feedback'] = '各要素をきちんと確認しましょう。'
            ans['data'] = sympy.matrix2numpy(sympy.Matrix(ls_sub.generator).transpose()).tolist() 
            answers.append(dict(ans))
            if sympy.Matrix(ls_sub.generator) != sympy.Matrix(ls.generator).transpose():
                ans['feedback'] = '今回の基底の計算に必要な行列は，列が1つのベクトルに対応する必要があります。'
                ans['data'] = ls_sub.generator
                answers.append(dict(ans))
            if _is_matrix:
                ans['feedback'] = '今回の基底の計算に必要な行列は，列が1つのベクトルに対応する必要があります。'
                ans['data'] = linear_algebra.thread_list([linear_algebra.partition_list(v, 2) for v in ls_sub.generator])
                answers.append(dict(ans))            
            answers = common.answer_union(answers)
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        _text = r'次のベクトルにより張られる部分空間の基底を，これらのベクトルの組として求めるとします。この計算を，行の基本変形により求める場合，どのような行列に掃き出しを行えばよいでしょうか。もっとも適切と思われる行列を選択してください。<br />'
        _text += self._str_list_vectors(quiz.data[0], quiz.data[1], quiz.data[2])
        return _text
    def answer_text(self, ans):
        return r'\( ' + sympy.latex(sympy.Matrix(ans['data']), mat_delim='', mat_str='pmatrix') + r' \)'
    def _str_list_vectors(self, ls, _is_polynomial, _is_matrix):
        _text = r'\( '
        for i in range(ls.num_gens):
            _text += ls.a_vector_as_str(ls.generator[i], is_latex_closure=False, is_polynomial=_is_polynomial, is_matrix=_is_matrix)
            if i < ls.num_gens - 1:
                _text += r',\;'        
        return _text + r' \)'


# In[5]:


if __name__ == "__main__":
    q = matrix_const_for_basis_comp()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=2)


# In[7]:


if __name__ == "__main__":
    pass
    #qz.save('matrix_const_for_basis_comp.xml')


# ## basis from reduced row echelon form

# In[53]:


class basis_from_reduced_row_echelon_form(core.Question):
    name = '掃き出し結果の行列から基底の組を構成する'
    def __init__(self, emin=-3, emax=3, dmin=2, dmax=5, rdefimin=0, rdefimax=3, rdunmin=0, rdunmax=1):
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
        # dummy
        self.ls_dummy = linear_algebra.LinearSpace()
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='掃き出し結果と基底', quiz_number=_quiz_number)
        ls = linear_algebra.LinearSpace()
        ls.set_dimension_range(self.dmin, self.dmax)
        ls.set_element_range(self.emin, self.emax)
        ls.set_rank_deficient_range(self.rdefimin, self.rdefimax)
        ls.set_redundant_range(self.rdunmin, self.rdunmax)
        ls.set_keep_integer(True)
        ls.set_pivot_one(True)
        ls.generate()
        while ls.num_gens < 2 or ls.num_disj < 2:
            ls.generate()
        _is_polynomial = False
        _is_matrix = False
        if ls.num_gens == 4 and random.random() < 0.5:
            _is_matrix = True
        elif random.random() < 0.5:
            _is_polynomial = True
        quiz.quiz_identifier = hash(ls)
        # 正答の選択肢の生成
        rref_matrix, rref_pivots = sympy.Matrix(ls.generator).rref()        
        quiz.data = [ls, rref_matrix, rref_pivots, _is_polynomial, _is_matrix]
        ans = { 'fraction': 100, 'data': [[sympy.matrix2numpy(sympy.Matrix(ls.generator).transpose()).tolist()[i] for i in rref_pivots], _is_polynomial, _is_matrix ]}
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        ls = quiz.data[0]
        rref_matrix = quiz.data[1]
        rref_pivots = quiz.data[2]
        _is_polynomial = quiz.data[3]
        _is_matrix = quiz.data[4]
        correct_data = [sympy.matrix2numpy(sympy.Matrix(ls.generator).transpose()).tolist()[i] for i in rref_pivots]
        if not _is_polynomial:
            ans['feedback'] = 'ベクトルの種類が変化しています。部分空間を生成しているベクトルの種類を確認しましょう。'
            ans['data'] = [correct_data, True, False ]
#            answers.append(dict(ans))
        if ls.num_gens == 4 and not _is_matrix:
            ans['feedback'] = 'ベクトルの種類が変化しています。部分空間を生成しているベクトルの種類を確認しましょう。'
            ans['data'] = [correct_data, False, True ]
#            answers.append(dict(ans))
        if _is_polynomial or _is_matrix:
            ans['feedback'] = 'ベクトルの種類が変化しています。部分空間を生成しているベクトルの種類を確認しましょう。'
            ans['data'] = [correct_data, False, False]
            answers.append(dict(ans))
        incorrect_data = [sympy.matrix2numpy(sympy.Matrix(ls.generator).transpose()).tolist()[i] for i in range(len(rref_pivots))]
        if correct_data != incorrect_data:
            ans['feedback'] = '基底は掃き出した行列の主成分に対応する元のベクトルになります。'
            ans['data'] = [incorrect_data, _is_polynomial, _is_matrix]
            answers.append(dict(ans))            
            if not _is_polynomial:
                ans['feedback'] = 'ベクトルの種類が変化しています。部分空間を生成しているベクトルの種類を確認しましょう。'
                ans['data'] = [incorrect_data, True, False ]
#                answers.append(dict(ans))
            if ls.num_gens == 4 and not _is_matrix:
                ans['feedback'] = 'ベクトルの種類が変化しています。部分空間を生成しているベクトルの種類を確認しましょう。'
                ans['data'] = [incorrect_data, False, True ]
#                answers.append(dict(ans))
            if _is_polynomial or _is_matrix:
                ans['feedback'] = 'ベクトルの種類が変化しています。部分空間を生成しているベクトルの種類を確認しましょう。'
                ans['data'] = [incorrect_data, False, False]
                answers.append(dict(ans))
        seq_max = ls.dimension
        for i in range(len(rref_pivots)):
            if rref_pivots[i] != i:
                seq_max = i
                break
        incorrect_data = [sympy.matrix2numpy(sympy.Matrix(ls.generator).transpose()).tolist()[i] for i in range(seq_max)]
        if correct_data != incorrect_data:
            ans['feedback'] = '基底は掃き出した行列の主成分に対応する元のベクトルになります。'
            ans['data'] = [incorrect_data, _is_polynomial, _is_matrix]
            answers.append(dict(ans))            
            if not _is_polynomial:
                ans['feedback'] = 'ベクトルの種類が変化しています。部分空間を生成しているベクトルの種類を確認しましょう。'
                ans['data'] = [incorrect_data, True, False ]
#                answers.append(dict(ans))
            if ls.num_gens == 4 and not _is_matrix:
                ans['feedback'] = 'ベクトルの種類が変化しています。部分空間を生成しているベクトルの種類を確認しましょう。'
                ans['data'] = [incorrect_data, False, True ]
#                answers.append(dict(ans))
            if _is_polynomial or _is_matrix:
                ans['feedback'] = 'ベクトルの種類が変化しています。部分空間を生成しているベクトルの種類を確認しましょう。'
                ans['data'] = [incorrect_data, False, False]
                answers.append(dict(ans))
        incorrect_data = [sympy.matrix2numpy(rref_matrix.transpose()).tolist()[i] for i in rref_pivots]
        if correct_data != incorrect_data:
            ans['feedback'] = '基底は掃き出した行列の主成分に対応する元のベクトルになります。'
            ans['data'] = [incorrect_data, _is_polynomial, _is_matrix]
            answers.append(dict(ans))            
        incorrect_data = [sympy.matrix2numpy(rref_matrix).tolist()[i] for i in range(rref_matrix.rank())]
        if correct_data != incorrect_data:
            ans['feedback'] = '基底は掃き出した行列の主成分に対応する元のベクトルになります。'
            ans['data'] = [incorrect_data, _is_polynomial, _is_matrix]
            answers.append(dict(ans))
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        ls = quiz.data[0]
        rref_matrix = quiz.data[1]
        rref_pivots = quiz.data[2]
        _is_polynomial = quiz.data[3]
        _is_matrix = quiz.data[4]
        _text = r'次のベクトル\( ' 
        for i in range(ls.dimension):
            _text += r'\vec{u_{' + str(i+1) + r'}}'
            if i < ls.dimension - 1:
                _text += r',\;'        
        _text += r' \)により張られる部分空間の基底を，これらのベクトルの組として求めるため，行列\( \begin{pmatrix}' 
        for i in range(ls.dimension):
            _text += r'\vec{u_{' + str(i+1) + r'}}'
            if i < ls.dimension - 1:
                _text += r'\;'        
        _text += r'\end{pmatrix} \)と列の線形関係が同一の数値行列に対して行の基本変形を行ったところ，次の行列\( A \)が得られた。この結果から部分空間の1組の基底として適切なものを選択してください。<br />'
        _text += self._str_list_vectors_trans_wu(ls, _is_polynomial, _is_matrix)
        _text += r'<br />'
        _text += r'\( A = ' + sympy.latex(rref_matrix, mat_delim='', mat_str='pmatrix') + r' \)'
        return _text
    def answer_text(self, ans):
        _text = r'\( '
        for i in range(len(ans['data'][0])):
            _text += self.ls_dummy.a_vector_as_str(ans['data'][0][i], is_latex_closure=False, is_polynomial=ans['data'][1], is_matrix=ans['data'][2])
            if i < len(ans['data'][0]) - 1:
                _text += r',\;'        
        return _text + r' \)'
    def _str_list_vectors_trans_wu(self, ls, _is_polynomial, _is_matrix):
        _text = r'\( '
        for i in range(ls.dimension):
            _text += r'\vec{u_{' + str(i+1) + r'}} = '
            _text += ls.a_vector_as_str(sympy.matrix2numpy(sympy.Matrix(ls.generator).transpose()).tolist()[i], is_latex_closure=False, is_polynomial=_is_polynomial, is_matrix=_is_matrix)
            if i < ls.dimension - 1:
                _text += r',\;'        
        return _text + r' \)'


# In[54]:


if __name__ == "__main__":
    q = basis_from_reduced_row_echelon_form()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=2)


# In[56]:


if __name__ == "__main__":
    pass
    #qz.save('basis_from_reduced_row_echelon_form.xml')


# ## basis computation for sub-space

# In[36]:


class basis_comp_for_subspace(core.Question):
    name = '様々な部分空間の基底の計算'
    def __init__(self, emin=-4, emax=4, dmin=2, dmax=3, rdefimin=1, rdefimax=2, rdunmin=1, rdunmax=3):
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
        self.ls_dummy = linear_algebra.LinearSpace()
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='部分空間の基底の計算', quiz_number=_quiz_number)
        is_polynomial = True if random.random() < 0.3 else False
        is_generator = True if is_polynomial or random.random() < 0.3 else False        
        ls = linear_algebra.LinearSpace()
        ls.set_dimension_range(self.dmin, self.dmax)
        ls.set_element_range(self.emin, self.emax)
        ls.set_rank_deficient_range(self.rdefimin, self.rdefimax)
        ls.set_redundant_range(self.rdunmin, self.rdunmax)
        ls.set_keep_integer(True)
        if not is_generator:
            ls.set_pivot_one(True)
            ls.set_pivot_rtl(True)
        ls.generate()
        rref_matrix, rref_pivots = sympy.Matrix(ls.generator).transpose().rref()
        if is_generator:
            avec = [ls.generator[i] for i in rref_pivots]
        else:
            avec = ls.basis
        quiz.data = [ls, is_polynomial, is_generator, rref_matrix, rref_pivots, avec]
        quiz.quiz_identifier = hash(ls)
        # 正答の選択肢の生成
        ans = { 'fraction': 100, 'data': [avec, is_polynomial] }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        ls = quiz.data[0]
        is_polynomial = quiz.data[1]
        is_generator = quiz.data[2]
        rref_matrix = quiz.data[3]
        rref_pivots = quiz.data[4]
        correct_answer = quiz.data[5]
        if is_polynomial:
            ans['feedback'] = '部分空間の種類が変化しています。ベクトルの種類を確認しましょう。'
            ans['data'] = [correct_answer, False ]
            answers.append(dict(ans))
        incorrect_answer = copy.deepcopy(correct_answer)
        r = random.randint(0, len(incorrect_answer) - 1)
        c = random.randint(0, len(incorrect_answer[0]) - 1)
        incorrect_answer[r][c] += random.choice([-2,-1,1,2])
        ans['feedback'] = 'ケアレスミスの可能性があります。よく要素を確認しましょう。'
        ans['data'] = [incorrect_answer, is_polynomial ]
        answers.append(dict(ans))
        if is_polynomial:
            ans['feedback'] = 'そもそも部分空間の種類が変化しています。ベクトルの種類を確認しましょう。'
            ans['data'] = [incorrect_answer, False ]
            answers.append(dict(ans))
        n = random.randint(1, ls.num_gens)
        while n == ls.num_disj:
            n = random.randint(1, ls.num_gens)
        incorrect_answer = random.sample(ls.generator, n)
        ans['feedback'] = '基底は部分空間を張り，かつ，線形独立である必要があります。'
        ans['data'] = [incorrect_answer, is_polynomial ]
        answers.append(dict(ans))
        if is_polynomial:
            ans['feedback'] = 'そもそも部分空間の種類が変化しています。ベクトルの種類を確認しましょう。'
            ans['data'] = [incorrect_answer, False ]
            answers.append(dict(ans))
        r = random.randint(0, len(incorrect_answer) - 1)
        c = random.randint(0, len(incorrect_answer[0]) - 1)
        incorrect_answer = copy.deepcopy(incorrect_answer)
        incorrect_answer[r][c] += random.choice([-2,-1,1,2])
        ans['feedback'] = '基底は部分空間を張り，かつ，線形独立である必要があります。'
        ans['data'] = [incorrect_answer, is_polynomial ]
        answers.append(dict(ans))
        if is_polynomial:
            ans['feedback'] = 'そもそも部分空間の種類が変化しています。ベクトルの種類を確認しましょう。'
            ans['data'] = [incorrect_answer, False ]
            answers.append(dict(ans))     
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)       
        return answers    
    def question_text(self, quiz):
        _text = '次の部分空間の基底として適切なものを選択してください。<br />'
        if quiz.data[1]:
            _text += quiz.data[0].str_as_spanned_space(is_polynomial=True)
        elif quiz.data[2]:
            _text += quiz.data[0].str_as_spanned_space(is_polynomial=False)
        else:
            _text += quiz.data[0].str_as_solution_space()
        return  _text
    def answer_text(self, ans):
        return self._str_list_vectors(ans['data'][0], ans['data'][1])
    def _str_list_vectors(self, vec, _is_polynomial):
        _text = r'\( \left\{'
        for i in range(len(vec)):
            _text += self.ls_dummy.a_vector_as_str(vec[i], is_latex_closure=False, is_polynomial=_is_polynomial, is_matrix=False)
            if i < len(vec) - 1:
                _text += r',\;'        
        return _text + r'\right\} \)'


# In[37]:


if __name__ == "__main__":
    q = basis_comp_for_subspace()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=2)


# In[38]:


if __name__ == "__main__":
    pass
    #qz.save('basis_comp_for_subspace.xml')


# ## dimension of sub-space intersection

# In[74]:


class dim_of_subspace_intersection(core.Question):
    name = '部分空間同士の共通部分の次元の計算'
    def __init__(self, emin=-4, emax=4, dmin=2, dmax=4, rdefimin=1, rdefimax=2, rdunmin=1, rdunmax=3):
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
        self.ls_dummy = linear_algebra.LinearSpace()
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='共通部分の次元の計算', quiz_number=_quiz_number)
        ls1 = linear_algebra.LinearSpace()
        ls1.set_dimension_range(self.dmin, self.dmax)
        ls1.set_element_range(self.emin, self.emax)
        ls1.set_rank_deficient_range(self.rdefimin, self.rdefimax)
        ls1.set_redundant_range(self.rdunmin, self.rdunmax)
        ls1.set_keep_integer(True)
        ls1.generate()
        while ls1.num_gens < 2:
            ls1.generate()     
        ls2 = linear_algebra.LinearSpace(ls1)
        ls2.set_dimension_range(ls1.dimension, ls1.dimension)
        _is_zero_ok = True if random.random() < 0.25 else False
        ls2.generate()
        r = sympy.Matrix(ls1.basis).rank() + sympy.Matrix(ls2.basis).rank() - sympy.Matrix([*ls1.basis, *ls2.basis]).rank()
        while r == 0 and not _is_zero_ok:
            ls2.generate()
            r = sympy.Matrix(ls1.basis).rank() + sympy.Matrix(ls2.basis).rank() - sympy.Matrix([*ls1.basis, *ls2.basis]).rank()            
        _is_polynomial = False
        _is_matrix = False
        if ls1.dimension == 4 and random.random() < 0.5:
            _is_matrix = True
        elif random.random() < 0.5:
            _is_polynomial = True
        # 正答の選択肢の生成
        quiz.data = [ls1, ls2, _is_polynomial, _is_matrix, r]
        quiz.quiz_identifier = hash(ls1) + hash(ls2)
        ans = { 'fraction': 100, 'data': r }
        quiz.answers.append(dict(ans))
        for i in range(0,ls1.dimension + 1):
            if i != r:
                ans = { 'fraction': 0, 'data': i, 'feedback': '共通部分の次元に成り立つ性質に基づき計算しましょう。' }
                quiz.answers.append(dict(ans))
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        # 正答を個別には作らないので何もしない
        pass
    def question_text(self, quiz):
        _text = r'1組の基底が'
        _text += self._str_list_vectors(quiz.data[0], quiz.data[2], quiz.data[3])
        _text += r'である部分空間\( U \)と，1組の基底が'
        _text += self._str_list_vectors(quiz.data[1], quiz.data[2], quiz.data[3])
        _text += r'である部分空間\( V \)の共通部分である\( U \cap V \)の次元を選択してください。<br />'
        return _text
    def answer_text(self, ans):
        return str(ans['data'])
    def _str_list_vectors(self, ls, _is_polynomial, _is_matrix):
        _text = r'\( \left\{'
        for i in range(ls.num_disj):
            _text += ls.a_vector_as_str(ls.basis[i], is_latex_closure=False, is_polynomial=_is_polynomial, is_matrix=_is_matrix)
            if i < ls.num_disj - 1:
                _text += r',\;'        
        return _text + r'\right\} \)'


# In[75]:


if __name__ == "__main__":
    q = dim_of_subspace_intersection()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=2)


# In[76]:


if __name__ == "__main__":
    pass
    #qz.save('dim_of_subspace_intersection.xml')


# ## select linear maps

# In[8]:


class select_linear_maps(core.Question):
    name = '線型写像をすべて選択'
    def __init__(self, emin=-5, emax=5, dmin=2, dmax=4, iddmin=0, iddmax=2, asize=6, camin=2, camax=4):
        # 生成する個々の数の範囲
        self.emin = emin
        self.emax = emax
        # 生成するベクトルの次元の範囲
        self.dmin = dmin
        self.dmax = dmax
        # 生成する写像の次元からのランク落ちの範囲
        self.iddmin = iddmin
        self.iddmax = iddmax
        # 選択肢の個数
        self.asize = asize
        # 正答選択肢の個数の範囲
        self.camin = camin
        self.camax = camax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='線型写像の峻別', single=False, quiz_number=_quiz_number)
        quiz.data = random.randint(self.camin, self.camax)
        quiz.quiz_identifier = hash(str(random.random()))
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        """
        This ignores the option value: size.
        """
        num_ca = quiz.data
        fra_ca = 100 / num_ca
        answers = []
        for i in range(num_ca):
            lm = linear_algebra.LinearMap()
            lm.set_element_range(self.emin, self.emax)
            lm.set_dimension_range(self.dmin, self.dmax)
            lm.set_image_dimension_deficient_range(self.iddmin, self.iddmax)
            lm.generate(is_linear=True, is_source_standard_basis=True, is_destination_standard_basis=True)
            ans = { 'fraction': fra_ca, 'data': lm }
            answers.append(dict(ans))
        return answers
    def incorrect_answers_generate(self, quiz, size=4):
        """
        This ignores the option value: size.
        """
        num_ica = self.asize - quiz.data
        answers = []
        for i in range(num_ica):
            lm = linear_algebra.LinearMap()
            lm.set_element_range(self.emin, self.emax)
            lm.set_dimension_range(self.dmin, self.dmax)
            lm.set_image_dimension_deficient_range(self.iddmin, self.iddmax)
            lm.generate(is_linear=False, is_source_standard_basis=True, is_destination_standard_basis=True)
            ans = { 'fraction': -100, 'data': lm, 'feedback': '線形写像の定義に基づき，線形性の成立を確認しましょう。線形写像の必要条件としては，ゼロベクトルをゼロベクトルへ写すことや，和との交換可能性があります。' }
            answers.append(dict(ans))
        return answers
    def question_text(self, quiz):
        _text = r'次の写像のうち，線型写像であるものをすべて選択してください。<br />'
        return _text
    def answer_text(self, ans):
        return ans['data'].str_map()


# In[9]:


if __name__ == "__main__":
    q = select_linear_maps()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=2)


# In[10]:


if __name__ == "__main__":
    pass
    #qz.save('select_linear_maps.xml')


# ## kernel of linear map

# In[13]:


class kernel_of_linear_map(core.Question):
    name = '数ベクトル空間の線形写像の核の次元と基底'
    def __init__(self, emin=-5, emax=5, dmin=2, dmax=4, iddmin=0, iddmax=2):
        # 生成する個々の数の範囲
        self.emin = emin
        self.emax = emax
        # 生成するベクトルの次元の範囲
        self.dmin = dmin
        self.dmax = dmax
        # 生成する写像の次元からのランク落ちの範囲
        self.iddmin = iddmin
        self.iddmax = iddmax
    def _rref_with_zero_fill(self, lm):
        _rref = []
        for i in range(lm.source_dim):
            if i in lm.pivot_positions:
                for j in range(len(lm.pivot_positions)):
                    if lm.pivot_positions[j] == i:
                        _rref.append(lm.rref[j])
            else:
                _rref.append([0 for j in range(lm.source_dim)])
        return _rref
    def _kernel_basis(self, lm):
        _rref = self._rref_with_zero_fill(lm)
        _basis_candidates = sympy.eye(lm.source_dim) - sympy.Matrix(_rref)
        _basis_candidates = sympy.matrix2numpy(_basis_candidates.transpose()).tolist()
        _basis = []
        for _vec in _basis_candidates:
            if not linear_algebra.is_zero_vector(_vec):
                _basis.append(_vec)
        _dim = len(_basis)
        if _basis == []:
            _basis = [[0 for i in range(lm.source_dim)]]
        return _dim,_basis
    def _kernel_basis_wrong_sign(self, lm):
        _rref = self._rref_with_zero_fill(lm)
        _basis_candidates_correct = sympy.eye(lm.source_dim) - sympy.Matrix(_rref)
        _basis_candidates_correct = sympy.matrix2numpy(_basis_candidates_correct).tolist()
        _basis_candidates = []
        for i in range(len(_basis_candidates_correct)):
            _basis_candidates.append([_basis_candidates_correct[i][j] if i == j else -_basis_candidates_correct[i][j] for j in range(len(_basis_candidates_correct))])
        _basis_candidates = sympy.Matrix(_basis_candidates)
        _basis_candidates = sympy.matrix2numpy(_basis_candidates.transpose()).tolist()
        _basis = []
        for _vec in _basis_candidates:
            if not linear_algebra.is_zero_vector(_vec):
                _basis.append(_vec)
        _dim = len(_basis)
        if _basis == []:
            _basis = [[0 for i in range(lm.source_dim)]]
        return _dim,_basis
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='核の次元と基底', quiz_number=_quiz_number)
        lm = linear_algebra.LinearMap()
        lm.set_element_range(self.emin, self.emax)
        lm.set_dimension_range(self.dmin, self.dmax)
        lm.set_image_dimension_deficient_range(self.iddmin, self.iddmax)
        lm.generate(is_linear=True, is_source_standard_basis=True, is_destination_standard_basis=True, is_force_symbolic=True, is_force_numeric=True)       
        quiz.quiz_identifier = hash(lm)
        # 正答の選択肢の生成
        correct_dim,correct_data = self._kernel_basis(lm)
        quiz.data = [lm, correct_dim, correct_data]        
        ans = { 'fraction': 100, 'data': [correct_dim, correct_data]}
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def _check_same_linear_space(self, correct_basis, incorrect_basis):
        if len(correct_basis[0]) != len(incorrect_basis[0]):
            return False
        elif linear_algebra.is_zero_vectors(correct_basis) and not linear_algebra.is_zero_vectors(incorrect_basis):
            return False
        elif linear_algebra.is_zero_vectors(incorrect_basis) and not linear_algebra.is_zero_vectors(correct_basis):
            return False
        else:
            return sympy.Matrix(correct_basis + incorrect_basis).rank() == len(correct_basis)
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        lm = quiz.data[0]
        correct_dim = quiz.data[1]
        correct_data = quiz.data[2]
        incorrect_dim = lm.source_dim
        if correct_dim != incorrect_dim:
            ans['feedback'] = '次元は基底を構成するベクトルの個数になります（零空間を除く）。空間の次元ではありません。'
            ans['data'] = [incorrect_dim, correct_data]
            answers.append(dict(ans))            
        incorrect_dim = lm.destination_dim
        if correct_dim != incorrect_dim:
            ans['feedback'] = '次元は基底を構成するベクトルの個数になります（零空間を除く）。空間の次元ではありません。'
            ans['data'] = [incorrect_dim, correct_data]
            answers.append(dict(ans))
        incorrect_data = [sympy.matrix2numpy(sympy.Matrix(lm.matrix).transpose()).tolist()[i] for i in lm.pivot_positions]
        incorrect_dim = len(incorrect_data)
        if len(incorrect_data) == 0:
            incorrect_dim = 0
            incorrect_data = [[0 for i in range(lm.destination_dim)]]
        if not self._check_same_linear_space(correct_data, incorrect_data):
            ans['feedback'] = '核は，線形写像によって零ベクトルに写るベクトルの集合です。像空間と取り違えています。'
            ans['data'] = [incorrect_dim, incorrect_data]
            answers.append(dict(ans))
            incorrect_dim = lm.source_dim
            if len(incorrect_data) != incorrect_dim:
                ans['feedback'] = '核は，線形写像によって零ベクトルに写るベクトルの集合です。次元の定義も確認しましょう。'
                ans['data'] = [incorrect_dim, incorrect_data]
                answers.append(dict(ans))            
            incorrect_dim = lm.destination_dim
            if len(incorrect_data) != incorrect_dim:
                ans['feedback'] = '核は，線形写像によって零ベクトルに写るベクトルの集合です。次元の定義も確認しましょう。'
                ans['data'] = [incorrect_dim, incorrect_data]
                answers.append(dict(ans))         
        incorrect_dim,incorrect_data = self._kernel_basis_wrong_sign(lm)
        if len(incorrect_data) == 0:
            incorrect_dim = 0
            incorrect_data = [[0 for i in range(lm.destination_dim)]]
        if not self._check_same_linear_space(correct_data, incorrect_data):
            ans['feedback'] = '同次線形方程式の解の構成を再確認しましょう。'
            ans['data'] = [incorrect_dim, incorrect_data]
            answers.append(dict(ans))
            incorrect_dim = lm.source_dim
            if len(incorrect_data) != incorrect_dim:
                ans['feedback'] = '同次線形方程式の解の構成を再確認しましょう。次元の定義も確認しましょう。'
                ans['data'] = [incorrect_dim, incorrect_data]
                answers.append(dict(ans))            
            incorrect_dim = lm.destination_dim
            if len(incorrect_data) != incorrect_dim:
                ans['feedback'] = '同次線形方程式の解の構成を再確認しましょう。次元の定義も確認しましょう。'
                ans['data'] = [incorrect_dim, incorrect_data]
                answers.append(dict(ans))         
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        lm = quiz.data[0]
        _text = r'次の線形写像の核の次元と1組の基底として適切なものを選択してください。<br />'
        _text += lm.str_map(is_latex_closure=True)
        _text += r'<br />なお，この線形写像に現れる行列の簡約な行列は，'
        if len(lm.rref) > 0:
            _text += r'\( ' + sympy.latex(sympy.Matrix(lm.rref), mat_delim='', mat_str='pmatrix') + r' \)となります。'
        else:
            _text += r'\( ' + sympy.latex(sympy.Matrix([[0 for i in range(lm.source_dim)]]), mat_delim='', mat_str='pmatrix') + r' \)となります。'
        return _text
    def answer_text(self, ans):
        _text = r'次元は\( ' + str(ans['data'][0]) + r' \)で，1組の基底は，\( \left\{'
        for i in range(len(ans['data'][1])):
            _text += sympy.latex(sympy.Matrix([ans['data'][1][i]]).transpose(), mat_delim='', mat_str='pmatrix')
            if i < len(ans['data'][1]) - 1:
                _text += r',\;'        
        return _text + r'\right\} \)'


# In[15]:


if __name__ == "__main__":
    q = kernel_of_linear_map()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=2)


# In[16]:


if __name__ == "__main__":
    pass
    #qz.save('kernel_of_linear_map.xml')


# ## image of linear map

# In[31]:


class image_of_linear_map(core.Question):
    name = '数ベクトル空間の線形写像の像の次元と基底'
    def __init__(self, emin=-5, emax=5, dmin=2, dmax=4, iddmin=0, iddmax=2):
        # 生成する個々の数の範囲
        self.emin = emin
        self.emax = emax
        # 生成するベクトルの次元の範囲
        self.dmin = dmin
        self.dmax = dmax
        # 生成する写像の次元からのランク落ちの範囲
        self.iddmin = iddmin
        self.iddmax = iddmax
    def _rref_with_zero_fill(self, lm):
        _rref = []
        for i in range(lm.source_dim):
            if i in lm.pivot_positions:
                for j in range(len(lm.pivot_positions)):
                    if lm.pivot_positions[j] == i:
                        _rref.append(lm.rref[j])
            else:
                _rref.append([0 for j in range(lm.source_dim)])
        return _rref
    def _kernel_basis(self, lm):
        _rref = self._rref_with_zero_fill(lm)
        _basis_candidates = sympy.eye(lm.source_dim) - sympy.Matrix(_rref)
        _basis_candidates = sympy.matrix2numpy(_basis_candidates.transpose()).tolist()
        _basis = []
        for _vec in _basis_candidates:
            if not linear_algebra.is_zero_vector(_vec):
                _basis.append(_vec)
        _dim = len(_basis)
        if _basis == []:
            _basis = [[0 for i in range(lm.source_dim)]]
        return _dim,_basis
    def _image_basis(self, lm):
        _basis = []
        _matrix = sympy.matrix2numpy(sympy.Matrix(lm.matrix).transpose()).tolist()
        for i in lm.pivot_positions:
            _basis.append(_matrix[i])
        _dim = len(_basis)
        if _basis == []:
            _basis = [[0 for i in range(lm.destination_dim)]]
        return _dim,_basis
    def _image_basis_wrong_pos(self, lm):
        _basis = []
        _matrix = sympy.matrix2numpy(sympy.Matrix(lm.matrix).transpose()).tolist()
        for i in range(len(_matrix)):
            if i not in lm.pivot_positions:
                _basis.append(_matrix[i])
        _dim = len(_basis)
        if _basis == []:
            _basis = [[0 for i in range(lm.source_dim)]]
        return _dim,_basis
    def _image_basis_wrong_pos2(self, lm):
        _basis = []
        _matrix = sympy.matrix2numpy(sympy.Matrix(lm.matrix).transpose()).tolist()
        for i in range(len(lm.pivot_positions)):
            _basis.append(_matrix[i])
        _dim = len(_basis)
        if _basis == []:
            _basis = [[0 for i in range(lm.source_dim)]]
        return _dim,_basis
    def _kernel_basis_wrong_sign(self, lm):
        _rref = self._rref_with_zero_fill(lm)
        _basis_candidates_correct = sympy.eye(lm.source_dim) - sympy.Matrix(_rref)
        _basis_candidates_correct = sympy.matrix2numpy(_basis_candidates_correct).tolist()
        _basis_candidates = []
        for i in range(len(_basis_candidates_correct)):
            _basis_candidates.append([_basis_candidates_correct[i][j] if i == j else -_basis_candidates_correct[i][j] for j in range(len(_basis_candidates_correct))])
        _basis_candidates = sympy.Matrix(_basis_candidates)
        _basis_candidates = sympy.matrix2numpy(_basis_candidates.transpose()).tolist()
        _basis = []
        for _vec in _basis_candidates:
            if not linear_algebra.is_zero_vector(_vec):
                _basis.append(_vec)
        _dim = len(_basis)
        if _basis == []:
            _basis = [[0 for i in range(lm.source_dim)]]
        return _dim,_basis
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='像の次元と基底', quiz_number=_quiz_number)
        lm = linear_algebra.LinearMap()
        lm.set_element_range(self.emin, self.emax)
        lm.set_dimension_range(self.dmin, self.dmax)
        lm.set_image_dimension_deficient_range(self.iddmin, self.iddmax)
        lm.generate(is_linear=True, is_source_standard_basis=True, is_destination_standard_basis=True, is_force_symbolic=True, is_force_numeric=True)       
        quiz.quiz_identifier = hash(lm)
        # 正答の選択肢の生成
        correct_dim,correct_data = self._image_basis(lm)
        quiz.data = [lm, correct_dim, correct_data]        
        ans = { 'fraction': 100, 'data': [correct_dim, correct_data]}
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def _check_same_linear_space(self, correct_basis, incorrect_basis):
        if len(correct_basis[0]) != len(incorrect_basis[0]):
            return False
        elif linear_algebra.is_zero_vectors(correct_basis) and not linear_algebra.is_zero_vectors(incorrect_basis):
            return False
        elif linear_algebra.is_zero_vectors(incorrect_basis) and not linear_algebra.is_zero_vectors(correct_basis):
            return False
        else:
            return sympy.Matrix(correct_basis + incorrect_basis).rank() == len(correct_basis) and len(correct_basis) == len(incorrect_basis)
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        lm = quiz.data[0]
        correct_dim = quiz.data[1]
        correct_data = quiz.data[2]
        incorrect_dim = lm.source_dim
        if correct_dim != incorrect_dim:
            ans['feedback'] = '次元は基底を構成するベクトルの個数になります（零空間を除く）。空間の次元ではありません。'
            ans['data'] = [incorrect_dim, correct_data]
            answers.append(dict(ans))            
        incorrect_dim = lm.destination_dim
        if correct_dim != incorrect_dim:
            ans['feedback'] = '次元は基底を構成するベクトルの個数になります（零空間を除く）。空間の次元ではありません。'
            ans['data'] = [incorrect_dim, correct_data]
            answers.append(dict(ans))  
        incorrect_dim,incorrect_data = self._image_basis_wrong_pos(lm)
        if not self._check_same_linear_space(correct_data, incorrect_data):
            ans['feedback'] = '一次独立な最大個数の計算と同じです。確認しましょう。'
            ans['data'] = [incorrect_dim, incorrect_data]
            answers.append(dict(ans))
            incorrect_dim = lm.source_dim
            if len(incorrect_data) != incorrect_dim:
                ans['feedback'] = '一次独立な最大個数の計算と同じです。確認しましょう。次元の定義も確認しましょう。'
                ans['data'] = [incorrect_dim, incorrect_data]
                answers.append(dict(ans))            
            incorrect_dim = lm.destination_dim
            if len(incorrect_data) != incorrect_dim:
                ans['feedback'] = '一次独立な最大個数の計算と同じです。確認しましょう。次元の定義も確認しましょう。'
                ans['data'] = [incorrect_dim, incorrect_data]
                answers.append(dict(ans))         
        incorrect_dim,incorrect_data = self._image_basis_wrong_pos2(lm)
        if not self._check_same_linear_space(correct_data, incorrect_data):
            ans['feedback'] = '一次独立な最大個数の計算と同じです。確認しましょう。'
            ans['data'] = [incorrect_dim, incorrect_data]
            answers.append(dict(ans))
            incorrect_dim = lm.source_dim
            if len(incorrect_data) != incorrect_dim:
                ans['feedback'] = '一次独立な最大個数の計算と同じです。確認しましょう。次元の定義も確認しましょう。'
                ans['data'] = [incorrect_dim, incorrect_data]
                answers.append(dict(ans))            
            incorrect_dim = lm.destination_dim
            if len(incorrect_data) != incorrect_dim:
                ans['feedback'] = '一次独立な最大個数の計算と同じです。確認しましょう。次元の定義も確認しましょう。'
                ans['data'] = [incorrect_dim, incorrect_data]
                answers.append(dict(ans))         
        incorrect_dim,incorrect_data = self._kernel_basis(lm)
        if len(incorrect_data) == 0:
            incorrect_dim = 0
            incorrect_data = [[0 for i in range(lm.destination_dim)]]
        if not self._check_same_linear_space(correct_data, incorrect_data):
            ans['feedback'] = '同次線形方程式の解とは異なります。像の定義を再確認しましょう。'
            ans['data'] = [incorrect_dim, incorrect_data]
            answers.append(dict(ans))
            incorrect_dim = lm.source_dim
            if len(incorrect_data) != incorrect_dim:
                ans['feedback'] = '同次線形方程式の解とは異なります。像の定義を再確認しましょう。次元の定義も確認しましょう。'
                ans['data'] = [incorrect_dim, incorrect_data]
                answers.append(dict(ans))            
            incorrect_dim = lm.destination_dim
            if len(incorrect_data) != incorrect_dim:
                ans['feedback'] = '同次線形方程式の解とは異なります。像の定義を再確認しましょう。次元の定義も確認しましょう。'
                ans['data'] = [incorrect_dim, incorrect_data]
                answers.append(dict(ans))         
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        lm = quiz.data[0]
        _text = r'次の線形写像の像の次元と1組の基底として適切なものを選択してください。<br />'
        _text += lm.str_map(is_latex_closure=True)
        _text += r'<br />なお，この線形写像に現れる行列の簡約な行列は，'
        if len(lm.rref) > 0:
            _text += r'\( ' + sympy.latex(sympy.Matrix(lm.rref), mat_delim='', mat_str='pmatrix') + r' \)となります。'
        else:
            _text += r'\( ' + sympy.latex(sympy.Matrix([[0 for i in range(lm.source_dim)]]), mat_delim='', mat_str='pmatrix') + r' \)となります。'
        return _text
    def answer_text(self, ans):
        _text = r'次元は\( ' + str(ans['data'][0]) + r' \)で，1組の基底は，\( \left\{'
        for i in range(len(ans['data'][1])):
            _text += sympy.latex(sympy.Matrix([ans['data'][1][i]]).transpose(), mat_delim='', mat_str='pmatrix')
            if i < len(ans['data'][1]) - 1:
                _text += r',\;'        
        return _text + r'\right\} \)'


# In[33]:


if __name__ == "__main__":
    q = image_of_linear_map()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=2)


# In[34]:


if __name__ == "__main__":
    pass
    #qz.save('image_of_linear_map.xml')


# ## matrix representation w.r.t. standard-standard basis

# In[25]:


class matrix_representation_standard_standard_basis(core.Question):
    name = '線形写像の標準基底の標準基底に関する表現行列'
    def __init__(self, emin=-5, emax=5, dmin=2, dmax=4, iddmin=0, iddmax=2):
        # 生成する個々の数の範囲
        self.emin = emin
        self.emax = emax
        # 生成するベクトルの次元の範囲
        self.dmin = dmin
        self.dmax = dmax
        # 生成する写像の次元からのランク落ちの範囲
        self.iddmin = iddmin
        self.iddmax = iddmax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='標準基底・標準基底に関する表現行列', quiz_number=_quiz_number)
        lm = linear_algebra.LinearMap()
        lm.set_element_range(self.emin, self.emax)
        lm.set_dimension_range(self.dmin, self.dmax)
        lm.set_image_dimension_deficient_range(self.iddmin, self.iddmax)
        lm.generate(is_linear=True, is_source_standard_basis=True, is_destination_standard_basis=True)       
        quiz.quiz_identifier = hash(lm)
        # 正答の選択肢の生成
        quiz.data = [lm, lm.matrix]        
        ans = { 'fraction': 100, 'data': lm.matrix}
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        lm = quiz.data[0]
        correct_data = quiz.data[1]
        incorrect_data = sympy.matrix2numpy(sympy.Matrix(correct_data).transpose()).tolist()
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '標準基底に関する表現行列は，標準基底の像を標準基底の線形結合で表したものです。基底の像が一致するか検算しましょう。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))                  
        incorrect_data = [_e for _e in reversed(sympy.matrix2numpy(sympy.Matrix(correct_data).transpose()).tolist())]
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '標準基底に関する表現行列は，標準基底の像を標準基底の線形結合で表したものです。基底の像が一致するか検算しましょう。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))            
        incorrect_data = [_e for _e in reversed(correct_data)]
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '標準基底に関する表現行列は，標準基底の像を標準基底の線形結合で表したものです。基底の像が一致するか検算しましょう。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))            
        incorrect_data = lm.source_basis
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '標準基底に関する表現行列は，標準基底の像を標準基底の線形結合で表したものです。基底そのものではありません。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))            
        incorrect_data = [_e for _e in reversed(lm.source_basis)]
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '標準基底に関する表現行列は，標準基底の像を標準基底の線形結合で表したものです。基底そのものではありません。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))            
        incorrect_data = lm.destination_basis
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '標準基底に関する表現行列は，標準基底の像を標準基底の線形結合で表したものです。基底そのものではありません。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))            
        incorrect_data = [_e for _e in reversed(lm.destination_basis)]
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '標準基底に関する表現行列は，標準基底の像を標準基底の線形結合で表したものです。基底そのものではありません。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))            
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        lm = quiz.data[0]
        _text = r'次の線形写像（\(U=' + lm.str_source_domain(is_latex_closure=False) + r'\)から'
        _text += r'\(V=' + lm.str_destination_domain(is_latex_closure=False) + r'\)への写像）の'
        _text += r'\(U\)の基底' + lm.str_source_basis() + r'の'
        _text += r'\(V\)の基底' + lm.str_destination_basis() + r'に関する表現行列を求めてください。'        
        _text += r'なお，この\(U\)の基底の像は，' + lm.str_image_of_source_basis()
        _text += r'となることを活用しても構いません。<br />'
        _text += lm.str_map(is_latex_closure=True)
        return _text
    def answer_text(self, ans):
        _text = r'\(' + sympy.latex(sympy.Matrix(ans['data']), mat_delim='', mat_str='pmatrix') + r'\)'
        return _text


# In[26]:


if __name__ == "__main__":
    q = matrix_representation_standard_standard_basis()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=2)


# In[5]:


if __name__ == "__main__":
    pass
    #qz.save('matrix_representation_standard_standard_basis.xml')


# ## matrix representation w.r.t. general-standard basis

# In[27]:


class matrix_representation_general_standard_basis(core.Question):
    name = '線形写像の一般基底の標準基底に関する表現行列'
    def __init__(self, emin=-5, emax=5, dmin=2, dmax=4, iddmin=0, iddmax=2):
        # 生成する個々の数の範囲
        self.emin = emin
        self.emax = emax
        # 生成するベクトルの次元の範囲
        self.dmin = dmin
        self.dmax = dmax
        # 生成する写像の次元からのランク落ちの範囲
        self.iddmin = iddmin
        self.iddmax = iddmax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='一般基底・標準基底に関する表現行列', quiz_number=_quiz_number)
        lm = linear_algebra.LinearMap()
        lm.set_element_range(self.emin, self.emax)
        lm.set_dimension_range(self.dmin, self.dmax)
        lm.set_image_dimension_deficient_range(self.iddmin, self.iddmax)
        lm.generate(is_linear=True, is_source_standard_basis=False, is_destination_standard_basis=True)       
        quiz.quiz_identifier = hash(lm)
        # 正答の選択肢の生成
        quiz.data = [lm, lm.representation_matrix]        
        ans = { 'fraction': 100, 'data': lm.representation_matrix}
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        lm = quiz.data[0]
        correct_data = quiz.data[1]
        incorrect_data = lm.matrix
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '標準基底に関する表現行列は，基底の像を標準基底の線形結合で表したものです。基底の像が一致するか検算しましょう。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))            
        incorrect_data = [_e for _e in reversed(sympy.matrix2numpy(sympy.Matrix(lm.matrix)).tolist())]
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '標準基底に関する表現行列は，基底の像を標準基底の線形結合で表したものです。基底の像が一致するか検算しましょう。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))       
        incorrect_data = sympy.matrix2numpy(sympy.Matrix(correct_data).transpose()).tolist()
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '標準基底に関する表現行列は，基底の像を標準基底の線形結合で表したものです。基底の像が一致するか検算しましょう。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))            
        incorrect_data = [_e for _e in reversed(sympy.matrix2numpy(sympy.Matrix(correct_data).transpose()).tolist())]
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '標準基底に関する表現行列は，基底の像を標準基底の線形結合で表したものです。基底の像が一致するか検算しましょう。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))            
        incorrect_data = [_e for _e in reversed(correct_data)]
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '標準基底に関する表現行列は，基底の像を標準基底の線形結合で表したものです。基底の像が一致するか検算しましょう。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))            
        incorrect_data = lm.source_basis
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '標準基底に関する表現行列は，基底の像を標準基底の線形結合で表したものです。基底そのものではありません。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))            
        incorrect_data = [_e for _e in reversed(lm.source_basis)]
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '標準基底に関する表現行列は，基底の像を標準基底の線形結合で表したものです。基底そのものではありません。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))            
        incorrect_data = lm.destination_basis
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '標準基底に関する表現行列は，基底の像を標準基底の線形結合で表したものです。基底そのものではありません。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))            
        incorrect_data = [_e for _e in reversed(lm.destination_basis)]
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '標準基底に関する表現行列は，基底の像を標準基底の線形結合で表したものです。基底そのものではありません。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))            
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        lm = quiz.data[0]
        _text = r'次の線形写像（\(U=' + lm.str_source_domain(is_latex_closure=False) + r'\)から'
        _text += r'\(V=' + lm.str_destination_domain(is_latex_closure=False) + r'\)への写像）の'
        _text += r'\(U\)の基底' + lm.str_source_basis() + r'の'
        _text += r'\(V\)の基底' + lm.str_destination_basis() + r'に関する表現行列を求めてください。'        
        _text += r'なお，この\(U\)の基底の像は，' + lm.str_image_of_source_basis()
        _text += r'となることを活用しても構いません。<br />'
        _text += lm.str_map(is_latex_closure=True)
        return _text
    def answer_text(self, ans):
        _text = r'\(' + sympy.latex(sympy.Matrix(ans['data']), mat_delim='', mat_str='pmatrix') + r'\)'
        return _text


# In[28]:


if __name__ == "__main__":
    q = matrix_representation_general_standard_basis()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[9]:


if __name__ == "__main__":
    pass
    #qz.save('matrix_representation_general_standard_basis.xml')


# ## change of basis matrix

# In[18]:


class change_of_basis_matrix(core.Question):
    name = '線形空間の基底の変換行列'
    def __init__(self, emin=-5, emax=5, dmin=2, dmax=3, iddmin=0, iddmax=2):
        # 生成する個々の数の範囲
        self.emin = emin
        self.emax = emax
        # 生成するベクトルの次元の範囲
        self.dmin = dmin
        self.dmax = dmax
        # 生成する写像の次元からのランク落ちの範囲
        self.iddmin = iddmin
        self.iddmax = iddmax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='基底の変換行列', quiz_number=_quiz_number)
        lm1 = linear_algebra.LinearMap()
        lm1.set_element_range(self.emin, self.emax)
        lm1.set_dimension_range(self.dmin, self.dmax)
        lm1.set_image_dimension_deficient_range(self.iddmin, self.iddmax)
        lm1.generate(is_linear=True, is_source_standard_basis=True, is_destination_standard_basis=False)
        lm2 = linear_algebra.LinearMap(lm1)
        lm2.generate_basis(is_source_standard_basis=True, is_destination_standard_basis=False)
        quiz.quiz_identifier = hash(lm1) + hash(lm2)
        # 正答の選択肢の生成
        correct_data = sympy.matrix2numpy(sympy.Matrix(lm1.destination_basis).transpose().inv()*sympy.Matrix(lm2.destination_basis).transpose()).tolist()
        quiz.data = [lm1, lm2, correct_data]
        ans = { 'fraction': 100, 'data': correct_data}
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        lm1 = quiz.data[0]
        lm2 = quiz.data[1]
        correct_data = quiz.data[2]
        incorrect_data = sympy.matrix2numpy(sympy.Matrix(correct_data).transpose()).tolist()
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '基底の変換行列は，変換元の基底を変換後の基底の線形結合で表したものです。計算方法を確認しましょう。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))            
        incorrect_data = [_e for _e in reversed(correct_data)]
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '基底の変換行列は，変換元の基底を変換後の基底の線形結合で表したものです。計算方法を確認しましょう。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))          
        incorrect_data = sympy.matrix2numpy(sympy.Matrix(lm2.destination_basis).transpose().inv()*sympy.Matrix(lm1.destination_basis).transpose()).tolist()
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '変換元と変換先を確認しましょう。逆になっています。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))          
        incorrect_data = sympy.matrix2numpy((sympy.Matrix(lm2.destination_basis).transpose().inv()*sympy.Matrix(lm1.destination_basis).transpose()).transpose()).tolist()
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '変換元と変換先を確認しましょう。逆になっています。計算方法も確認しましょう。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))
        incorrect_data = sympy.matrix2numpy(sympy.Matrix(lm1.destination_basis).transpose()).tolist()
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '基底の変換行列は，変換元の基底を変換後の基底の線形結合で表したものです。基底そのものではありません。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))          
        incorrect_data = sympy.matrix2numpy(sympy.Matrix(lm2.destination_basis).transpose()).tolist()
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '基底の変換行列は，変換元の基底を変換後の基底の線形結合で表したものです。基底そのものではありません。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))          
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        lm1 = quiz.data[0]
        lm2 = quiz.data[1]
        _text = r'線形空間\(U\)の2つの基底\(W_1=' + lm1.str_destination_basis(is_latex_closure=False) + r'\)と'
        _text += r'\(W_2=' + lm2.str_destination_basis(is_latex_closure=False) + r'\)に関して，'
        _text += r'\(W_2\)による座標を\(W_1\)による座標表示に直す基底の変換行列を選んでください。'
        _text += r'なお，以下の関係式を活用してください。<br />'
        _text += r'\(' + sympy.latex(sympy.Matrix(lm1.destination_basis), mat_delim='', mat_str='pmatrix') + r'^{-1}'
        _text += r'=' + sympy.latex(sympy.Matrix(lm1.destination_basis).inv(), mat_delim='', mat_str='pmatrix') + r',\;'
        _text += sympy.latex(sympy.Matrix(lm1.destination_basis).transpose(), mat_delim='', mat_str='pmatrix') + r'^{-1}'
        _text += r'=' + sympy.latex(sympy.Matrix(lm1.destination_basis).transpose().inv(), mat_delim='', mat_str='pmatrix') + r',\;'
        _text += sympy.latex(sympy.Matrix(lm2.destination_basis), mat_delim='', mat_str='pmatrix') + r'^{-1}'
        _text += r'=' + sympy.latex(sympy.Matrix(lm2.destination_basis).inv(), mat_delim='', mat_str='pmatrix') + r',\;'
        _text += sympy.latex(sympy.Matrix(lm2.destination_basis).transpose(), mat_delim='', mat_str='pmatrix') + r'^{-1}'
        _text += r'=' + sympy.latex(sympy.Matrix(lm2.destination_basis).transpose().inv(), mat_delim='', mat_str='pmatrix') + r'\)'
        return _text
    def answer_text(self, ans):
        _text = r'\(' + sympy.latex(sympy.Matrix(ans['data']), mat_delim='', mat_str='pmatrix') + r'\)'
        return _text


# In[19]:


if __name__ == "__main__":
    q = change_of_basis_matrix()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[20]:


if __name__ == "__main__":
    pass
    #qz.save('change_of_basis_matrix.xml')


# ## matrix representation w.r.t. general-general numeric basis

# In[6]:


class matrix_representation_general_general_numeric_basis(core.Question):
    name = '線形写像の一般基底の一般基底に関する表現行列（数ベクトル空間）'
    def __init__(self, emin=-5, emax=5, dmin=2, dmax=4, iddmin=0, iddmax=2):
        # 生成する個々の数の範囲
        self.emin = emin
        self.emax = emax
        # 生成するベクトルの次元の範囲
        self.dmin = dmin
        self.dmax = dmax
        # 生成する写像の次元からのランク落ちの範囲
        self.iddmin = iddmin
        self.iddmax = iddmax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='一般基底・一般基底に関する表現行列（数ベクトル空間）', quiz_number=_quiz_number)
        lm = linear_algebra.LinearMap()
        lm.set_element_range(self.emin, self.emax)
        lm.set_dimension_range(self.dmin, self.dmax)
        lm.set_image_dimension_deficient_range(self.iddmin, self.iddmax)
        lm.generate(is_linear=True, is_force_numeric=True, is_source_standard_basis=False, is_destination_standard_basis=False)       
        quiz.quiz_identifier = hash(lm)
        # 正答の選択肢の生成
        quiz.data = [lm, lm.representation_matrix]        
        ans = { 'fraction': 100, 'data': lm.representation_matrix}
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        lm = quiz.data[0]
        correct_data = quiz.data[1]
        incorrect_data = lm.matrix
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '一般基底に関する表現行列は，標準基底での表現行列を求め，基底の変換行列を活用しましょう。検算もしましょう。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))            
        incorrect_data = [_e for _e in reversed(sympy.matrix2numpy(sympy.Matrix(lm.matrix)).tolist())]
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '一般基底に関する表現行列は，標準基底での表現行列を求め，基底の変換行列を活用しましょう。検算もしましょう。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))            
        incorrect_data = sympy.matrix2numpy(sympy.Matrix(correct_data).transpose()).tolist()
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '一般基底に関する表現行列は，標準基底での表現行列を求め，基底の変換行列を活用しましょう。検算もしましょう。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))            
        incorrect_data = [_e for _e in reversed(sympy.matrix2numpy(sympy.Matrix(correct_data).transpose()).tolist())]
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '一般基底に関する表現行列は，標準基底での表現行列を求め，基底の変換行列を活用しましょう。検算もしましょう。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))            
        incorrect_data = [_e for _e in reversed(correct_data)]
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '一般基底に関する表現行列は，標準基底での表現行列を求め，基底の変換行列を活用しましょう。検算もしましょう。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))            
        incorrect_data = lm.source_basis
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '一般基底に関する表現行列は，標準基底での表現行列を求め，基底の変換行列を活用しましょう。基底そのものではありません。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))            
        incorrect_data = [_e for _e in reversed(lm.source_basis)]
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '一般基底に関する表現行列は，標準基底での表現行列を求め，基底の変換行列を活用しましょう。基底そのものではありません。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))            
        incorrect_data = lm.destination_basis
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '一般基底に関する表現行列は，標準基底での表現行列を求め，基底の変換行列を活用しましょう。基底そのものではありません。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))            
        incorrect_data = [_e for _e in reversed(lm.destination_basis)]
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '一般基底に関する表現行列は，標準基底での表現行列を求め，基底の変換行列を活用しましょう。基底そのものではありません。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))            
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        lm = quiz.data[0]
        _text = r'次の線形写像（\(U=' + lm.str_source_domain(is_latex_closure=False) + r'\)から'
        _text += r'\(V=' + lm.str_destination_domain(is_latex_closure=False) + r'\)への写像）の'
        _text += r'\(U\)の基底' + lm.str_source_basis() + r'の'
        _text += r'\(V\)の基底' + lm.str_destination_basis() + r'に関する表現行列を求めてください。<br />'        
        _text += lm.str_map(is_latex_closure=True)
        _text += r'<br />なお，この\(U\)の標準基底の像は，' + lm.str_image_of_standard_basis()
        _text += r'となることと次の関係式を活用しても構いません。<br />'
        _text += r'\(' + sympy.latex(sympy.Matrix(lm.source_basis), mat_delim='', mat_str='pmatrix') + r'^{-1}'
        _text += r'=' + sympy.latex(sympy.Matrix(lm.source_basis).inv(), mat_delim='', mat_str='pmatrix') + r',\;'
        _text += sympy.latex(sympy.Matrix(lm.source_basis).transpose(), mat_delim='', mat_str='pmatrix') + r'^{-1}'
        _text += r'=' + sympy.latex(sympy.Matrix(lm.source_basis).transpose().inv(), mat_delim='', mat_str='pmatrix') + r',\;'
        _text += sympy.latex(sympy.Matrix(lm.destination_basis), mat_delim='', mat_str='pmatrix') + r'^{-1}'
        _text += r'=' + sympy.latex(sympy.Matrix(lm.destination_basis).inv(), mat_delim='', mat_str='pmatrix') + r',\;'
        _text += sympy.latex(sympy.Matrix(lm.destination_basis).transpose(), mat_delim='', mat_str='pmatrix') + r'^{-1}'
        _text += r'=' + sympy.latex(sympy.Matrix(lm.destination_basis).transpose().inv(), mat_delim='', mat_str='pmatrix') + r'\)'
        return _text
    def answer_text(self, ans):
        _text = r'\(' + sympy.latex(sympy.Matrix(ans['data']), mat_delim='', mat_str='pmatrix') + r'\)'
        return _text


# In[7]:


if __name__ == "__main__":
    q = matrix_representation_general_general_numeric_basis()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[8]:


if __name__ == "__main__":
    pass
    #qz.save('matrix_representation_general_general_numeric_basis.xml')


# ## matrix representation w.r.t. general-general basis

# In[3]:


class matrix_representation_general_general_basis(core.Question):
    name = '線形写像の一般基底の一般基底に関する表現行列'
    def __init__(self, emin=-5, emax=5, dmin=2, dmax=4, iddmin=0, iddmax=2):
        # 生成する個々の数の範囲
        self.emin = emin
        self.emax = emax
        # 生成するベクトルの次元の範囲
        self.dmin = dmin
        self.dmax = dmax
        # 生成する写像の次元からのランク落ちの範囲
        self.iddmin = iddmin
        self.iddmax = iddmax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='一般基底・一般基底に関する表現行列', quiz_number=_quiz_number)
        lm = linear_algebra.LinearMap()
        lm.set_element_range(self.emin, self.emax)
        lm.set_dimension_range(self.dmin, self.dmax)
        lm.set_image_dimension_deficient_range(self.iddmin, self.iddmax)
        lm.generate(is_linear=True, is_source_standard_basis=False, is_destination_standard_basis=False)       
        quiz.quiz_identifier = hash(lm)
        # 正答の選択肢の生成
        quiz.data = [lm, lm.representation_matrix]        
        ans = { 'fraction': 100, 'data': lm.representation_matrix}
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        lm = quiz.data[0]
        correct_data = quiz.data[1]
        incorrect_data = lm.matrix
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '一般基底に関する表現行列は，標準基底での表現行列を求め，基底の変換行列を活用しましょう。検算もしましょう。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))            
        incorrect_data = [_e for _e in reversed(sympy.matrix2numpy(sympy.Matrix(lm.matrix)).tolist())]
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '一般基底に関する表現行列は，標準基底での表現行列を求め，基底の変換行列を活用しましょう。検算もしましょう。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))            
        incorrect_data = sympy.matrix2numpy(sympy.Matrix(correct_data).transpose()).tolist()
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '一般基底に関する表現行列は，標準基底での表現行列を求め，基底の変換行列を活用しましょう。検算もしましょう。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))            
        incorrect_data = [_e for _e in reversed(sympy.matrix2numpy(sympy.Matrix(correct_data).transpose()).tolist())]
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '一般基底に関する表現行列は，標準基底での表現行列を求め，基底の変換行列を活用しましょう。検算もしましょう。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))            
        incorrect_data = [_e for _e in reversed(correct_data)]
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '一般基底に関する表現行列は，標準基底での表現行列を求め，基底の変換行列を活用しましょう。検算もしましょう。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))            
        incorrect_data = lm.source_basis
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '一般基底に関する表現行列は，標準基底での表現行列を求め，基底の変換行列を活用しましょう。基底そのものではありません。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))            
        incorrect_data = [_e for _e in reversed(lm.source_basis)]
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '一般基底に関する表現行列は，標準基底での表現行列を求め，基底の変換行列を活用しましょう。基底そのものではありません。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))            
        incorrect_data = lm.destination_basis
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '一般基底に関する表現行列は，標準基底での表現行列を求め，基底の変換行列を活用しましょう。基底そのものではありません。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))            
        incorrect_data = [_e for _e in reversed(lm.destination_basis)]
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '一般基底に関する表現行列は，標準基底での表現行列を求め，基底の変換行列を活用しましょう。基底そのものではありません。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))            
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        lm = quiz.data[0]
        _text = r'次の線形写像（\(U=' + lm.str_source_domain(is_latex_closure=False) + r'\)から'
        _text += r'\(V=' + lm.str_destination_domain(is_latex_closure=False) + r'\)への写像）の'
        _text += r'\(U\)の基底' + lm.str_source_basis() + r'の'
        _text += r'\(V\)の基底' + lm.str_destination_basis() + r'に関する表現行列を求めてください。<br />'        
        _text += lm.str_map(is_latex_closure=True)
        _text += r'<br />なお，この\(U\)の標準基底の像は，' + lm.str_image_of_standard_basis()
        _text += r'となることと次の関係式を活用しても構いません。<br />'
        _text += r'\(' + sympy.latex(sympy.Matrix(lm.source_basis), mat_delim='', mat_str='pmatrix') + r'^{-1}'
        _text += r'=' + sympy.latex(sympy.Matrix(lm.source_basis).inv(), mat_delim='', mat_str='pmatrix') + r',\;'
        _text += sympy.latex(sympy.Matrix(lm.source_basis).transpose(), mat_delim='', mat_str='pmatrix') + r'^{-1}'
        _text += r'=' + sympy.latex(sympy.Matrix(lm.source_basis).transpose().inv(), mat_delim='', mat_str='pmatrix') + r',\;'
        _text += sympy.latex(sympy.Matrix(lm.destination_basis), mat_delim='', mat_str='pmatrix') + r'^{-1}'
        _text += r'=' + sympy.latex(sympy.Matrix(lm.destination_basis).inv(), mat_delim='', mat_str='pmatrix') + r',\;'
        _text += sympy.latex(sympy.Matrix(lm.destination_basis).transpose(), mat_delim='', mat_str='pmatrix') + r'^{-1}'
        _text += r'=' + sympy.latex(sympy.Matrix(lm.destination_basis).transpose().inv(), mat_delim='', mat_str='pmatrix') + r'\)'
        return _text
    def answer_text(self, ans):
        _text = r'\(' + sympy.latex(sympy.Matrix(ans['data']), mat_delim='', mat_str='pmatrix') + r'\)'
        return _text


# In[4]:


if __name__ == "__main__":
    q = matrix_representation_general_general_basis()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[5]:


if __name__ == "__main__":
    pass
    #qz.save('matrix_representation_general_general_basis.xml')


# ## matrix representation w.r.t. general numeric basis

# In[6]:


class matrix_representation_general_numeric_basis(core.Question):
    name = '線形変換の一般基底に関する表現行列（数ベクトル空間）'
    def __init__(self, emin=-5, emax=5, dmin=2, dmax=4, iddmin=0, iddmax=2):
        # 生成する個々の数の範囲
        self.emin = emin
        self.emax = emax
        # 生成するベクトルの次元の範囲
        self.dmin = dmin
        self.dmax = dmax
        # 生成する写像の次元からのランク落ちの範囲
        self.iddmin = iddmin
        self.iddmax = iddmax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='一般基底に関する表現行列（数ベクトル空間・線形変換）', quiz_number=_quiz_number)
        lm = linear_algebra.LinearMap()
        lm.set_element_range(self.emin, self.emax)
        lm.set_dimension_range(self.dmin, self.dmax)
        lm.set_image_dimension_deficient_range(self.iddmin, self.iddmax)
        lm.generate(is_linear=True, is_source_standard_basis=False, is_destination_standard_basis=False, is_force_numeric=True, is_force_same_space=True)       
        quiz.quiz_identifier = hash(lm)
        # 正答の選択肢の生成
        quiz.data = [lm, lm.representation_matrix]        
        ans = { 'fraction': 100, 'data': lm.representation_matrix}
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        lm = quiz.data[0]
        correct_data = quiz.data[1]
        incorrect_data = lm.matrix
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '一般基底に関する表現行列は，標準基底での表現行列を求め，基底の変換行列を活用しましょう。検算もしましょう。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))            
        incorrect_data = [_e for _e in reversed(sympy.matrix2numpy(sympy.Matrix(lm.matrix)).tolist())]
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '一般基底に関する表現行列は，標準基底での表現行列を求め，基底の変換行列を活用しましょう。検算もしましょう。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))            
        incorrect_data = sympy.matrix2numpy(sympy.Matrix(correct_data).transpose()).tolist()
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '一般基底に関する表現行列は，標準基底での表現行列を求め，基底の変換行列を活用しましょう。検算もしましょう。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))            
        incorrect_data = [_e for _e in reversed(sympy.matrix2numpy(sympy.Matrix(correct_data).transpose()).tolist())]
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '一般基底に関する表現行列は，標準基底での表現行列を求め，基底の変換行列を活用しましょう。検算もしましょう。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))            
        incorrect_data = [_e for _e in reversed(correct_data)]
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '一般基底に関する表現行列は，標準基底での表現行列を求め，基底の変換行列を活用しましょう。検算もしましょう。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))            
        incorrect_data = lm.source_basis
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '一般基底に関する表現行列は，標準基底での表現行列を求め，基底の変換行列を活用しましょう。基底そのものではありません。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))            
        incorrect_data = [_e for _e in reversed(lm.source_basis)]
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '一般基底に関する表現行列は，標準基底での表現行列を求め，基底の変換行列を活用しましょう。基底そのものではありません。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))            
        incorrect_data = lm.destination_basis
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '一般基底に関する表現行列は，標準基底での表現行列を求め，基底の変換行列を活用しましょう。基底そのものではありません。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))            
        incorrect_data = [_e for _e in reversed(lm.destination_basis)]
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '一般基底に関する表現行列は，標準基底での表現行列を求め，基底の変換行列を活用しましょう。基底そのものではありません。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))            
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        lm = quiz.data[0]
        _text = r'次の線形変換の基底' + lm.str_source_basis() + r'に関する表現行列を求めてください。<br />'        
        _text += lm.str_map(is_latex_closure=True)
        _text += r'<br />なお，このベクトル空間の標準基底の像は，' + lm.str_image_of_standard_basis()
        _text += r'となることと次の関係式を活用しても構いません。<br />'
        _text += r'\(' + sympy.latex(sympy.Matrix(lm.destination_basis), mat_delim='', mat_str='pmatrix') + r'^{-1}'
        _text += r'=' + sympy.latex(sympy.Matrix(lm.destination_basis).inv(), mat_delim='', mat_str='pmatrix') + r',\;'
        _text += sympy.latex(sympy.Matrix(lm.destination_basis).transpose(), mat_delim='', mat_str='pmatrix') + r'^{-1}'
        _text += r'=' + sympy.latex(sympy.Matrix(lm.destination_basis).transpose().inv(), mat_delim='', mat_str='pmatrix') + r'\)'
        return _text
    def answer_text(self, ans):
        _text = r'\(' + sympy.latex(sympy.Matrix(ans['data']), mat_delim='', mat_str='pmatrix') + r'\)'
        return _text


# In[7]:


if __name__ == "__main__":
    q = matrix_representation_general_numeric_basis()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[8]:


if __name__ == "__main__":
    pass
    #qz.save('matrix_representation_general_numeric_basis.xml')


# ## matrix representation w.r.t. general basis

# In[3]:


class matrix_representation_general_basis(core.Question):
    name = '線形変換の一般基底に関する表現行列'
    def __init__(self, emin=-5, emax=5, dmin=2, dmax=4, iddmin=0, iddmax=2):
        # 生成する個々の数の範囲
        self.emin = emin
        self.emax = emax
        # 生成するベクトルの次元の範囲
        self.dmin = dmin
        self.dmax = dmax
        # 生成する写像の次元からのランク落ちの範囲
        self.iddmin = iddmin
        self.iddmax = iddmax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='一般基底に関する表現行列（線形変換）', quiz_number=_quiz_number)
        lm = linear_algebra.LinearMap()
        lm.set_element_range(self.emin, self.emax)
        lm.set_dimension_range(self.dmin, self.dmax)
        lm.set_image_dimension_deficient_range(self.iddmin, self.iddmax)
        lm.generate(is_linear=True, is_source_standard_basis=False, is_destination_standard_basis=False, is_force_same_space=True)       
        quiz.quiz_identifier = hash(lm)
        # 正答の選択肢の生成
        quiz.data = [lm, lm.representation_matrix]        
        ans = { 'fraction': 100, 'data': lm.representation_matrix}
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        lm = quiz.data[0]
        correct_data = quiz.data[1]
        incorrect_data = lm.matrix
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '一般基底に関する表現行列は，標準基底での表現行列を求め，基底の変換行列を活用しましょう。検算もしましょう。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))            
        incorrect_data = [_e for _e in reversed(sympy.matrix2numpy(sympy.Matrix(lm.matrix)).tolist())]
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '一般基底に関する表現行列は，標準基底での表現行列を求め，基底の変換行列を活用しましょう。検算もしましょう。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))            
        incorrect_data = sympy.matrix2numpy(sympy.Matrix(correct_data).transpose()).tolist()
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '一般基底に関する表現行列は，標準基底での表現行列を求め，基底の変換行列を活用しましょう。検算もしましょう。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))            
        incorrect_data = [_e for _e in reversed(sympy.matrix2numpy(sympy.Matrix(correct_data).transpose()).tolist())]
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '一般基底に関する表現行列は，標準基底での表現行列を求め，基底の変換行列を活用しましょう。検算もしましょう。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))            
        incorrect_data = [_e for _e in reversed(correct_data)]
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '一般基底に関する表現行列は，標準基底での表現行列を求め，基底の変換行列を活用しましょう。検算もしましょう。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))            
        incorrect_data = lm.source_basis
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '一般基底に関する表現行列は，標準基底での表現行列を求め，基底の変換行列を活用しましょう。基底そのものではありません。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))            
        incorrect_data = [_e for _e in reversed(lm.source_basis)]
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '一般基底に関する表現行列は，標準基底での表現行列を求め，基底の変換行列を活用しましょう。基底そのものではありません。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))            
        incorrect_data = lm.destination_basis
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '一般基底に関する表現行列は，標準基底での表現行列を求め，基底の変換行列を活用しましょう。基底そのものではありません。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))            
        incorrect_data = [_e for _e in reversed(lm.destination_basis)]
        if sympy.Matrix(correct_data) != sympy.Matrix(incorrect_data):
            ans['feedback'] = '一般基底に関する表現行列は，標準基底での表現行列を求め，基底の変換行列を活用しましょう。基底そのものではありません。'
            ans['data'] = incorrect_data
            answers.append(dict(ans))            
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        lm = quiz.data[0]
        _text = r'次の線形変換の基底' + lm.str_source_basis() + r'に関する表現行列を求めてください。<br />'        
        _text += lm.str_map(is_latex_closure=True)
        _text += r'<br />なお，このベクトル空間の標準基底の像は，' + lm.str_image_of_standard_basis()
        _text += r'となることと次の関係式を活用しても構いません。<br />'
        _text += r'\(' + sympy.latex(sympy.Matrix(lm.destination_basis), mat_delim='', mat_str='pmatrix') + r'^{-1}'
        _text += r'=' + sympy.latex(sympy.Matrix(lm.destination_basis).inv(), mat_delim='', mat_str='pmatrix') + r',\;'
        _text += sympy.latex(sympy.Matrix(lm.destination_basis).transpose(), mat_delim='', mat_str='pmatrix') + r'^{-1}'
        _text += r'=' + sympy.latex(sympy.Matrix(lm.destination_basis).transpose().inv(), mat_delim='', mat_str='pmatrix') + r'\)'
        return _text
    def answer_text(self, ans):
        _text = r'\(' + sympy.latex(sympy.Matrix(ans['data']), mat_delim='', mat_str='pmatrix') + r'\)'
        return _text


# In[4]:


if __name__ == "__main__":
    q = matrix_representation_general_basis()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[5]:


if __name__ == "__main__":
    pass
    #qz.save('matrix_representation_general_basis.xml')


# # All the questions

# In[34]:


questions_str = ['general_set_operations', 'general_set_notations', 'same_subspace_recognition', 'a_subspace_recognition', 
                'representable_vector_by_linear_combination', 'nontrivial_linear_relation_vector', 
                'linearly_independent_check_equation', 'linearly_independent_check_equation_with_basis', 
                'rref_and_linearly_dependent_relation', 'maximum_linearly_independent_set', 
                'maximum_linearly_independent_set_general_case', 'matrix_const_for_basis_comp', 
                'basis_from_reduced_row_echelon_form', 'basis_comp_for_subspace', 
                'dim_of_subspace_intersection', 'select_linear_maps', 
                'kernel_of_linear_map', 'image_of_linear_map', 'matrix_representation_standard_standard_basis', 
                'matrix_representation_general_standard_basis', 'change_of_basis_matrix',
                'matrix_representation_general_general_numeric_basis', 
                'matrix_representation_general_general_basis', 
                'matrix_representation_general_numeric_basis', 'matrix_representation_general_basis']
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




