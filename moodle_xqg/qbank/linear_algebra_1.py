#!/usr/bin/env python
# coding: utf-8

# Moodle XML Question Generator, (C) 2020 Kosaku Nagasaka (Kobe University)

# ## モジュールファイルの生成

# In[1]:


if __name__ == "__main__":
    import subprocess
    subprocess.run(['jupyter', 'nbconvert', '--to', 'python', '_linear_algebra_1.ipynb','--output','linear_algebra_1.py'])


# # Linear Algebra 1 (matrix and linear equation)

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
import random
import IPython
import itertools
import copy


# ## NOTES

# - Any matrix will be translated in the latex format with "pmatrix". Please translate it with your preferable environment (e.g. bmatrix) by the option of the top level function.
# - Any vector symbol will be translated in the latex format with "\vec". Please translate it with your preferable command (e.g. \boldsymbol) by the option of the top level function.

# ## matrix notation in R^(2x2)

# In[3]:


class matrix_notation_R22(core.Question):
    name = '行列の記法(2行2列の行列)'
    _problem_types = ['element', 'row', 'col']
    def __init__(self, emin=-3, emax=3):
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='行列の記法(2x2)', quiz_number=_quiz_number)
        _type = random.choice(self._problem_types)
        _rows = 2
        _cols = 2
        _elem = [random.randint(0,_rows), random.randint(0,_cols)]
        _matA = sympy.Matrix([[random.randint(self.elem_min, self.elem_max) for i in range(_cols)] for j in range(_rows)])
        if _type == 'element':
            if _elem[0] < _rows and _elem[1] < _cols:
                _ans = _matA[_elem[0],_elem[1]]
            else:
                _ans = None
        elif _type == 'row':
            if _elem[0] < _rows:
                _ans = _matA[_elem[0],:]
            else:
                _ans = None
        else:
            if _elem[1] < _cols:
                _ans = _matA[:,_elem[1]]
            else:
                _ans = None
        quiz.quiz_identifier = hash(str(_matA) + str(_type))
        # 正答の選択肢の生成
        quiz.data = [_type, _elem, _rows, _cols, _matA, _ans]
        ans = { 'fraction': 100, 'data': _ans }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_type, _elem, _rows, _cols, _matA, _ans] = quiz.data
        answers = common.answer_union(answers)
        for r in range(_rows):
            incorrect_ans = _matA[r,:]
            if incorrect_ans != _ans:
                ans['feedback'] = r'行と列の意味，そして行数と列数を考え，指定された部分が存在するのかよく考えましょう。'
                ans['data'] = incorrect_ans
                answers.append(dict(ans))
        for c in range(_cols):
            incorrect_ans = _matA[:,c]
            if incorrect_ans != _ans:
                ans['feedback'] = r'行と列の意味，そして行数と列数を考え，指定された部分が存在するのかよく考えましょう。'
                ans['data'] = incorrect_ans
                answers.append(dict(ans))
        for r in range(_rows):
            for c in range(_cols):
                incorrect_ans = _matA[r,c]
                if incorrect_ans != _ans:
                    ans['feedback'] = r'行と列の意味，そして行数と列数を考え，指定された部分が存在するのかよく考えましょう。'
                    ans['data'] = incorrect_ans
                    answers.append(dict(ans))
        answers = common.answer_union(answers)
        if type(_ans) is not str:
            if len(answers) >= size - 1:
                answers = random.sample(answers,k=size)
            ans['feedback'] = r'存在しますので，きちんと確認をしてください。'
            ans['data'] = None
            answers.append(dict(ans))
        else:
            if len(answers) >= size:
                return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_type, _elem, _rows, _cols, _matA, _ans] = quiz.data
        _text = r'次の行列の'
        if _type == 'element':
            _text += '(' + str(_elem[0]+1) + ',' + str(_elem[1]+1) + r')成分'
        elif _type == 'row':
            _text += r'第' + str(_elem[0]+1) + r'行'
        else:
            _text += r'第' + str(_elem[1]+1) + r'列'        
        _text += r'を選択してください。<br />'
        _text += r'\( ' + sympy.latex(_matA, mat_delim='', mat_str='pmatrix') + r' \)'
        return _text
    def answer_text(self, ans):
        if ans['data'] is None:
            return r'存在しない。'
        else:
            return r'\( ' +  sympy.latex(ans['data'], mat_delim='', mat_str='pmatrix') + r' \)'


# In[4]:


if __name__ == "__main__":
    q = matrix_notation_R22()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[102]:


if __name__ == "__main__":
    pass
    #qz.save('matrix_notation_R22.xml')


# ## matrix-scalar computation in R^(2x2)

# In[5]:


class matrix_scalar_computation_R22(core.Question):
    name = '行列のスカラー倍(2行2列の整数行列)'
    def __init__(self, emin=-3, emax=3):
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='行列のスカラー倍(2x2)', quiz_number=_quiz_number)
        mc = linear_algebra_next_generation.MatrixComputations()
        mc.set_element_range(self.elem_min,self.elem_max)
        mc.generate("ms_22")
        quiz.quiz_identifier = hash(mc)
        # 正答の選択肢の生成
        quiz.data = mc
        ans = { 'fraction': 100, 'data': mc.get_value() }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        mc = quiz.data
        incorrect_value = mc.get_incorrect_value(incorrect_type='forgot_rest')
        if mc.get_value() != incorrect_value:
            ans['feedback'] = r'計算が途中で終わっているようです。丁寧に確認しましょう。'
            ans['data'] = incorrect_value
            answers.append(dict(ans))
        incorrect_value = mc.get_incorrect_value(incorrect_type='forgot_next_cols')
        if mc.get_value() != incorrect_value:
            ans['feedback'] = r'計算が行列の途中で終わっているようです。丁寧に確認しましょう。'
            ans['data'] = incorrect_value
            answers.append(dict(ans))
        incorrect_value = mc.get_incorrect_value(incorrect_type='forgot_next_rows')
        if mc.get_value() != incorrect_value:
            ans['feedback'] = r'計算が行列の途中で終わっているようです。丁寧に確認しましょう。'
            ans['data'] = incorrect_value
            answers.append(dict(ans))
        incorrect_value = mc.get_incorrect_value(incorrect_type='computation_miss')
        if mc.get_value() != incorrect_value:
            ans['feedback'] = r'計算が間違っているようです。丁寧に確認しましょう。'
            ans['data'] = incorrect_value
            answers.append(dict(ans))
        incorrect_value = mc.get_incorrect_value(incorrect_type='mul_instead_dot')
        if mc.get_value() != incorrect_value:
            ans['feedback'] = r'行列の計算の規則を誤って覚えています。確認しましょう。'
            ans['data'] = incorrect_value
            answers.append(dict(ans))
        answers = common.answer_union(answers)
        while len(answers) < size:
            incorrect_value = mc.get_incorrect_value(incorrect_type='mixed')
            if mc.get_value() != incorrect_value:
                ans['feedback'] = r'様々な点で間違っています。計算方法を再確認しましょう。'
                ans['data'] = incorrect_value
                answers.append(dict(ans))            
            answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        mc = quiz.data
        _text = r'次の計算をしてください。<br />'
        _text += mc.str_expression(is_latex_closure=True,is_symbolic=False)
        return _text
    def answer_text(self, ans):
        return r'\( ' +  sympy.latex(ans['data'], mat_delim='', mat_str='pmatrix') + r' \)'


# In[6]:


if __name__ == "__main__":
    q = matrix_scalar_computation_R22()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[7]:


if __name__ == "__main__":
    pass
    #qz.save('matrix_scalar_computation_R22.xml')


# ## matrix-matrix addition in R^(2x2)

# In[8]:


class matrix_matrix_addition_R22(core.Question):
    name = '行列同士の和(2行2列の整数行列)'
    def __init__(self, emin=-3, emax=3):
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='行列同士の和(2x2)', quiz_number=_quiz_number)
        mc = linear_algebra_next_generation.MatrixComputations()
        mc.set_element_range(self.elem_min,self.elem_max)
        mc.generate("m_m_22")
        quiz.quiz_identifier = hash(mc)
        # 正答の選択肢の生成
        quiz.data = mc
        ans = { 'fraction': 100, 'data': mc.get_value() }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        mc = quiz.data
        incorrect_value = mc.get_incorrect_value(incorrect_type='forgot_rest')
        if mc.get_value() != incorrect_value:
            ans['feedback'] = r'計算が途中で終わっているようです。丁寧に確認しましょう。'
            ans['data'] = incorrect_value
            answers.append(dict(ans))
        incorrect_value = mc.get_incorrect_value(incorrect_type='forgot_next_cols')
        if mc.get_value() != incorrect_value:
            ans['feedback'] = r'計算が行列の途中で終わっているようです。丁寧に確認しましょう。'
            ans['data'] = incorrect_value
            answers.append(dict(ans))
        incorrect_value = mc.get_incorrect_value(incorrect_type='forgot_next_rows')
        if mc.get_value() != incorrect_value:
            ans['feedback'] = r'計算が行列の途中で終わっているようです。丁寧に確認しましょう。'
            ans['data'] = incorrect_value
            answers.append(dict(ans))
        incorrect_value = mc.get_incorrect_value(incorrect_type='computation_miss')
        if mc.get_value() != incorrect_value:
            ans['feedback'] = r'計算が間違っているようです。丁寧に確認しましょう。'
            ans['data'] = incorrect_value
            answers.append(dict(ans))
        incorrect_value = mc.get_incorrect_value(incorrect_type='mul_instead_dot')
        if mc.get_value() != incorrect_value:
            ans['feedback'] = r'行列の計算の規則を誤って覚えています。確認しましょう。'
            ans['data'] = incorrect_value
            answers.append(dict(ans))
        answers = common.answer_union(answers)
        while len(answers) < size:
            incorrect_value = mc.get_incorrect_value(incorrect_type='mixed')
            if mc.get_value() != incorrect_value:
                ans['feedback'] = r'様々な点で間違っています。計算方法を再確認しましょう。'
                ans['data'] = incorrect_value
                answers.append(dict(ans))            
            answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        mc = quiz.data
        _text = r'次の計算をしてください。<br />'
        _text += mc.str_expression(is_latex_closure=True,is_symbolic=False)
        return _text
    def answer_text(self, ans):
        return r'\( ' +  sympy.latex(ans['data'], mat_delim='', mat_str='pmatrix') + r' \)'


# In[9]:


if __name__ == "__main__":
    q = matrix_matrix_addition_R22()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[10]:


if __name__ == "__main__":
    pass
    #qz.save('matrix_matrix_addition_R22.xml')


# ## matrix-vector multiplication in R^(2x2)

# In[11]:


class matrix_vector_multiplication_R22(core.Question):
    name = '行列とベクトルの積(2行2列の整数行列)'
    def __init__(self, emin=-3, emax=3):
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='行列とベクトルの積(2x2)', quiz_number=_quiz_number)
        mc = linear_algebra_next_generation.MatrixComputations()
        mc.set_element_range(self.elem_min,self.elem_max)
        mc.generate("mv_22")
        quiz.quiz_identifier = hash(mc)
        # 正答の選択肢の生成
        quiz.data = mc
        ans = { 'fraction': 100, 'data': mc.get_value() }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        mc = quiz.data
        incorrect_value = mc.get_incorrect_value(incorrect_type='forgot_rest')
        if mc.get_value() != incorrect_value:
            ans['feedback'] = r'計算が途中で終わっているようです。丁寧に確認しましょう。'
            ans['data'] = incorrect_value
            answers.append(dict(ans))
        incorrect_value = mc.get_incorrect_value(incorrect_type='forgot_next_cols')
        if mc.get_value() != incorrect_value:
            ans['feedback'] = r'計算が行列の途中で終わっているようです。丁寧に確認しましょう。'
            ans['data'] = incorrect_value
            answers.append(dict(ans))
        incorrect_value = mc.get_incorrect_value(incorrect_type='forgot_next_rows')
        if mc.get_value() != incorrect_value:
            ans['feedback'] = r'計算が行列の途中で終わっているようです。丁寧に確認しましょう。'
            ans['data'] = incorrect_value
            answers.append(dict(ans))
        incorrect_value = mc.get_incorrect_value(incorrect_type='computation_miss')
        if mc.get_value() != incorrect_value:
            ans['feedback'] = r'計算が間違っているようです。丁寧に確認しましょう。'
            ans['data'] = incorrect_value
            answers.append(dict(ans))
        incorrect_value = mc.get_incorrect_value(incorrect_type='mul_instead_dot')
        if mc.get_value() != incorrect_value:
            ans['feedback'] = r'行列の計算の規則を誤って覚えています。確認しましょう。'
            ans['data'] = incorrect_value
            answers.append(dict(ans))
        answers = common.answer_union(answers)
        while len(answers) < size:
            incorrect_value = mc.get_incorrect_value(incorrect_type='mixed')
            if mc.get_value() != incorrect_value:
                ans['feedback'] = r'様々な点で間違っています。計算方法を再確認しましょう。'
                ans['data'] = incorrect_value
                answers.append(dict(ans))            
            answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        mc = quiz.data
        _text = r'次の計算をしてください。<br />'
        _text += mc.str_expression(is_latex_closure=True,is_symbolic=False)
        return _text
    def answer_text(self, ans):
        return r'\( ' +  sympy.latex(ans['data'], mat_delim='', mat_str='pmatrix') + r' \)'


# In[12]:


if __name__ == "__main__":
    q = matrix_vector_multiplication_R22()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[13]:


if __name__ == "__main__":
    pass
    #qz.save('matrix_vector_multiplication_R22.xml')


# ## matrix-matrix multiplication in R^(2x2)

# In[14]:


class matrix_matrix_multiplication_R22(core.Question):
    name = '行列同士の積(2行2列の整数行列)'
    def __init__(self, emin=-3, emax=3):
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='行列同士の積(2x2)', quiz_number=_quiz_number)
        mc = linear_algebra_next_generation.MatrixComputations()
        mc.set_element_range(self.elem_min,self.elem_max)
        mc.generate("mm_22")
        quiz.quiz_identifier = hash(mc)
        # 正答の選択肢の生成
        quiz.data = mc
        ans = { 'fraction': 100, 'data': mc.get_value() }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        mc = quiz.data
        incorrect_value = mc.get_incorrect_value(incorrect_type='forgot_rest')
        if mc.get_value() != incorrect_value:
            ans['feedback'] = r'計算が途中で終わっているようです。丁寧に確認しましょう。'
            ans['data'] = incorrect_value
            answers.append(dict(ans))
        incorrect_value = mc.get_incorrect_value(incorrect_type='forgot_next_cols')
        if mc.get_value() != incorrect_value:
            ans['feedback'] = r'計算が行列の途中で終わっているようです。丁寧に確認しましょう。'
            ans['data'] = incorrect_value
            answers.append(dict(ans))
        incorrect_value = mc.get_incorrect_value(incorrect_type='forgot_next_rows')
        if mc.get_value() != incorrect_value:
            ans['feedback'] = r'計算が行列の途中で終わっているようです。丁寧に確認しましょう。'
            ans['data'] = incorrect_value
            answers.append(dict(ans))
        incorrect_value = mc.get_incorrect_value(incorrect_type='computation_miss')
        if mc.get_value() != incorrect_value:
            ans['feedback'] = r'計算が間違っているようです。丁寧に確認しましょう。'
            ans['data'] = incorrect_value
            answers.append(dict(ans))
        incorrect_value = mc.get_incorrect_value(incorrect_type='mul_instead_dot')
        if mc.get_value() != incorrect_value:
            ans['feedback'] = r'行列の計算の規則を誤って覚えています。確認しましょう。'
            ans['data'] = incorrect_value
            answers.append(dict(ans))
        answers = common.answer_union(answers)
        while len(answers) < size:
            incorrect_value = mc.get_incorrect_value(incorrect_type='mixed')
            if mc.get_value() != incorrect_value:
                ans['feedback'] = r'様々な点で間違っています。計算方法を再確認しましょう。'
                ans['data'] = incorrect_value
                answers.append(dict(ans))            
            answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        mc = quiz.data
        _text = r'次の計算をしてください。<br />'
        _text += mc.str_expression(is_latex_closure=True,is_symbolic=False)
        return _text
    def answer_text(self, ans):
        return r'\( ' +  sympy.latex(ans['data'], mat_delim='', mat_str='pmatrix') + r' \)'


# In[15]:


if __name__ == "__main__":
    q = matrix_matrix_multiplication_R22()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[16]:


if __name__ == "__main__":
    pass
    #qz.save('matrix_matrix_multiplication_R22.xml')


# ## matrix-expression computation in R^(2x2)

# In[23]:


class matrix_expression_computation_R22(core.Question):
    name = '行列の式の計算(2行2列の整数行列)'
    def __init__(self, emin=-3, emax=3):
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='行列の式の計算(2x2)', quiz_number=_quiz_number)
        mc = linear_algebra_next_generation.MatrixComputations()
        mc.set_element_range(self.elem_min,self.elem_max)
        mc.generate("msvm_22")
        quiz.quiz_identifier = hash(mc)
        # 正答の選択肢の生成
        quiz.data = mc
        ans = { 'fraction': 100, 'data': mc.get_value() }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        mc = quiz.data
        incorrect_value = mc.get_incorrect_value(incorrect_type='forgot_rest')
        if mc.get_value() != incorrect_value:
            ans['feedback'] = r'計算が途中で終わっているようです。丁寧に確認しましょう。'
            ans['data'] = incorrect_value
            answers.append(dict(ans))
        incorrect_value = mc.get_incorrect_value(incorrect_type='forgot_next_cols')
        if mc.get_value() != incorrect_value:
            ans['feedback'] = r'計算が行列の途中で終わっているようです。丁寧に確認しましょう。'
            ans['data'] = incorrect_value
            answers.append(dict(ans))
        incorrect_value = mc.get_incorrect_value(incorrect_type='forgot_next_rows')
        if mc.get_value() != incorrect_value:
            ans['feedback'] = r'計算が行列の途中で終わっているようです。丁寧に確認しましょう。'
            ans['data'] = incorrect_value
            answers.append(dict(ans))
        incorrect_value = mc.get_incorrect_value(incorrect_type='computation_miss')
        if mc.get_value() != incorrect_value:
            ans['feedback'] = r'計算が間違っているようです。丁寧に確認しましょう。'
            ans['data'] = incorrect_value
            answers.append(dict(ans))
        incorrect_value = mc.get_incorrect_value(incorrect_type='mul_instead_dot')
        if mc.get_value() != incorrect_value:
            ans['feedback'] = r'行列の計算の規則を誤って覚えています。確認しましょう。'
            ans['data'] = incorrect_value
            answers.append(dict(ans))
        answers = common.answer_union(answers)
        while len(answers) < size:
            incorrect_value = mc.get_incorrect_value(incorrect_type='mixed')
            if mc.get_value() != incorrect_value:
                ans['feedback'] = r'様々な点で間違っています。計算方法を再確認しましょう。'
                ans['data'] = incorrect_value
                answers.append(dict(ans))            
            answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        mc = quiz.data
        _text = r'次の行列等に対して，' + mc.str_expression(is_latex_closure=True,is_symbolic=True) + r'を計算してください。<br />'
        _text += mc.str_definition(is_latex_closure=True)
        return _text
    def answer_text(self, ans):
        return r'\( ' +  sympy.latex(ans['data'], mat_delim='', mat_str='pmatrix') + r' \)'


# In[24]:


if __name__ == "__main__":
    q = matrix_expression_computation_R22()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[25]:


if __name__ == "__main__":
    pass
    #qz.save('matrix_expression_computation_R22.xml')


# ## linear map image of line in R^(2x2)

# In[78]:


class linear_map_image_of_line_R22(core.Question):
    name = '直線の線形変換での像(R2の平面)'
    def __init__(self, emin=-3, emax=3):
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='直線の線形変換での像(平面)', quiz_number=_quiz_number)
        _abs_max = sympy.Max(sympy.Abs(self.elem_min), sympy.Abs(self.elem_max))
        while True:
            matA = sympy.Matrix([[0,0],[0,0]])
            while matA.rank() < 2 or (matA[0,0] == 0 and matA[0,1] == 0):
                matA = sympy.Matrix([[random.randint(self.elem_min,self.elem_max) for j in range(2)] for i in range(2)])
            alpha = linear_algebra_next_generation.nonzero_randint(self.elem_min,self.elem_max)
            beta = random.randint(self.elem_min,self.elem_max)
            while matA[0,1] != 0 and matA[0,1]*alpha + matA[0,0] == 0:
                alpha = linear_algebra_next_generation.nonzero_randint(self.elem_min,self.elem_max)
            new_alpha = sympy.Rational(matA[1,0]+matA[1,1]*alpha, matA[0,0]+matA[0,1]*alpha)
            new_beta = matA[1,1]*beta-sympy.Rational(matA[0,1]*beta*(matA[1,0]+matA[1,1]*alpha), matA[0,0]+matA[0,1]*alpha)
            if sympy.Abs(sympy.denom(new_alpha)) <= 2*_abs_max and sympy.Abs(sympy.numer(new_alpha)) <= 2*_abs_max and sympy.Abs(sympy.denom(new_beta)) <= 2*_abs_max and sympy.Abs(sympy.numer(new_beta)) <= 2*_abs_max:
                break
        quiz.quiz_identifier = hash(str(matA)+str(alpha)+str(beta))
        # 正答の選択肢の生成
        quiz.data = [matA,alpha,beta,new_alpha,new_beta]
        ans = { 'fraction': 100, 'data': [new_alpha,new_beta] }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [matA,alpha,beta,new_alpha,new_beta] = quiz.data
        if matA[0,0]+matA[0,1]*(-alpha) != 0:
            incorrect_alpha = sympy.Rational(matA[1,0]+matA[1,1]*(-alpha), matA[0,0]+matA[0,1]*(-alpha))
            incorrect_beta = matA[1,1]*beta-sympy.Rational(matA[0,1]*beta*(matA[1,0]+matA[1,1]*(-alpha)), matA[0,0]+matA[0,1]*(-alpha))
            if new_alpha != incorrect_alpha or new_beta != incorrect_beta:
                ans['feedback'] = r'直線の式の符号を取り違えている可能性があります。確認しましょう。'
                ans['data'] = [incorrect_alpha,incorrect_beta]
                answers.append(dict(ans))
        incorrect_alpha = sympy.Rational(matA[1,0]+matA[1,1]*alpha, matA[0,0]+matA[0,1]*alpha)
        incorrect_beta = matA[1,1]*(-beta)-sympy.Rational(matA[0,1]*(-beta)*(matA[1,0]+matA[1,1]*alpha), matA[0,0]+matA[0,1]*alpha)
        if new_alpha != incorrect_alpha or new_beta != incorrect_beta:
            ans['feedback'] = r'直線の式の符号を取り違えている可能性があります。確認しましょう。'
            ans['data'] = [incorrect_alpha,incorrect_beta]
            answers.append(dict(ans))
        incorrect_alpha = matA[0,0]*alpha + matA[0,1]*beta
        incorrect_beta = matA[1,0]*alpha + matA[1,1]*beta
        if new_alpha != incorrect_alpha or new_beta != incorrect_beta:
            ans['feedback'] = r'きちんとベクトルでの直線の式に直して，像の計算を行いましょう。'
            ans['data'] = [incorrect_alpha,incorrect_beta]
            answers.append(dict(ans))
        incorrect_alpha = alpha
        incorrect_beta = beta
        if new_alpha != incorrect_alpha or new_beta != incorrect_beta:
            ans['feedback'] = r'きちんとベクトルでの直線の式に直して，像の計算を行いましょう。'
            ans['data'] = [incorrect_alpha,incorrect_beta]
            answers.append(dict(ans))
        answers = common.answer_union(answers)
        while len(answers) < size:
            incorrect_alpha = new_alpha + random.choice([-1,0,1])
            incorrect_beta = new_beta + random.choice([-1,0,1])
            if new_alpha != incorrect_alpha or new_beta != incorrect_beta:
                ans['feedback'] = r'計算方法を再確認しましょう。'
                ans['data'] = [incorrect_alpha,incorrect_beta]
                answers.append(dict(ans))            
            answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    @staticmethod
    def _str_line(alpha, beta):
        _x,_y = sympy.symbols('x,y')
        return sympy.latex(sympy.Eq(_y, alpha*_x + beta), order='lex',long_frac_ratio=1)    
    def question_text(self, quiz):
        [matA,alpha,beta,new_alpha,new_beta] = quiz.data
        _text = r'行列\(' + sympy.latex(matA, mat_delim='', mat_str='pmatrix') + r'\)'
        _text += r'による線形変換での直線\(' + self._str_line(alpha, beta) + r'\)'      
        _text += r'の像を求めてください。'
        return _text
    def answer_text(self, ans):
        [new_alpha,new_beta] = ans['data']
        return r'\( ' + self._str_line(new_alpha, new_beta) + r' \)'


# In[79]:


if __name__ == "__main__":
    q = linear_map_image_of_line_R22()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[80]:


if __name__ == "__main__":
    pass
    #qz.save('linear_map_image_of_line_R22.xml')


# ## matrix notation in R^(mxn)

# In[96]:


class matrix_notation_Rmn(core.Question):
    name = '行列の記法(一般サイズの行列)'
    _problem_types = ['element', 'row', 'col']
    def __init__(self, dmin=2, dmax=5, emin=-3, emax=3):
        # 生成する行列のサイズの範囲
        self.dim_min = dmin
        self.dim_max = dmax
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='行列の記法(mxn)', quiz_number=_quiz_number)
        _type = random.choice(self._problem_types)
        _rows = random.randint(self.dim_min, self.dim_max)
        _cols = random.randint(self.dim_min, self.dim_max)
        _elem = [random.randint(0,_rows), random.randint(0,_cols)]
        _matA = sympy.Matrix([[random.randint(self.elem_min, self.elem_max) for i in range(_cols)] for j in range(_rows)])
        if _type == 'element':
            if _elem[0] < _rows and _elem[1] < _cols:
                _ans = _matA[_elem[0],_elem[1]]
            else:
                _ans = None
        elif _type == 'row':
            if _elem[0] < _rows:
                _ans = _matA[_elem[0],:]
            else:
                _ans = None
        else:
            if _elem[1] < _cols:
                _ans = _matA[:,_elem[1]]
            else:
                _ans = None
        quiz.quiz_identifier = hash(str(_matA) + str(_type))
        # 正答の選択肢の生成
        quiz.data = [_type, _elem, _rows, _cols, _matA, _ans]
        ans = { 'fraction': 100, 'data': _ans }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_type, _elem, _rows, _cols, _matA, _ans] = quiz.data
        answers = common.answer_union(answers)
        for r in range(_rows):
            incorrect_ans = _matA[r,:]
            if incorrect_ans != _ans:
                ans['feedback'] = r'行と列の意味，そして行数と列数を考え，指定された部分が存在するのかよく考えましょう。'
                ans['data'] = incorrect_ans
                answers.append(dict(ans))
        for c in range(_cols):
            incorrect_ans = _matA[:,c]
            if incorrect_ans != _ans:
                ans['feedback'] = r'行と列の意味，そして行数と列数を考え，指定された部分が存在するのかよく考えましょう。'
                ans['data'] = incorrect_ans
                answers.append(dict(ans))
        for r in range(_rows):
            for c in range(_cols):
                incorrect_ans = _matA[r,c]
                if incorrect_ans != _ans:
                    ans['feedback'] = r'行と列の意味，そして行数と列数を考え，指定された部分が存在するのかよく考えましょう。'
                    ans['data'] = incorrect_ans
                    answers.append(dict(ans))
        answers = common.answer_union(answers)
        if type(_ans) is not str:
            if len(answers) >= size - 1:
                answers = random.sample(answers,k=size)
            ans['feedback'] = r'存在しますので，きちんと確認をしてください。'
            ans['data'] = None
            answers.append(dict(ans))
        else:
            if len(answers) >= size:
                return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_type, _elem, _rows, _cols, _matA, _ans] = quiz.data
        _text = r'次の行列の'
        if _type == 'element':
            _text += '(' + str(_elem[0]+1) + ',' + str(_elem[1]+1) + r')成分'
        elif _type == 'row':
            _text += r'第' + str(_elem[0]+1) + r'行'
        else:
            _text += r'第' + str(_elem[1]+1) + r'列'        
        _text += r'を選択してください。<br />'
        _text += r'\( ' + sympy.latex(_matA, mat_delim='', mat_str='pmatrix') + r' \)'
        return _text
    def answer_text(self, ans):
        if ans['data'] is None:
            return r'存在しない。'
        else:
            return r'\( ' +  sympy.latex(ans['data'], mat_delim='', mat_str='pmatrix') + r' \)'


# In[98]:


if __name__ == "__main__":
    q = matrix_notation_Rmn()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[99]:


if __name__ == "__main__":
    pass
    #qz.save('matrix_notation_Rmn.xml')


# ## matrix-scalar computation in R^(mxn)

# In[117]:


class matrix_scalar_computation_Rmn(core.Question):
    name = '行列のスカラー倍(m行n列の整数行列)'
    def __init__(self, dmin=3, dmax=4, emin=-3, emax=3):
        # 生成する行列のサイズの範囲
        self.dim_min = dmin
        self.dim_max = dmax
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='行列のスカラー倍(mxn)', quiz_number=_quiz_number)
        mc = linear_algebra_next_generation.MatrixComputations()
        mc.set_dimension_range(self.dim_min,self.dim_max)
        mc.set_element_range(self.elem_min,self.elem_max)
        mc.generate("ms_mn")
        quiz.quiz_identifier = hash(mc)
        # 正答の選択肢の生成
        quiz.data = mc
        ans = { 'fraction': 100, 'data': mc.get_value() }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        mc = quiz.data
        incorrect_value = mc.get_incorrect_value(incorrect_type='forgot_rest')
        if mc.get_value() != incorrect_value:
            ans['feedback'] = r'計算が途中で終わっているようです。丁寧に確認しましょう。'
            ans['data'] = incorrect_value
            answers.append(dict(ans))
        incorrect_value = mc.get_incorrect_value(incorrect_type='forgot_next_cols')
        if mc.get_value() != incorrect_value:
            ans['feedback'] = r'計算が行列の途中で終わっているようです。丁寧に確認しましょう。'
            ans['data'] = incorrect_value
            answers.append(dict(ans))
        incorrect_value = mc.get_incorrect_value(incorrect_type='forgot_next_rows')
        if mc.get_value() != incorrect_value:
            ans['feedback'] = r'計算が行列の途中で終わっているようです。丁寧に確認しましょう。'
            ans['data'] = incorrect_value
            answers.append(dict(ans))
        incorrect_value = mc.get_incorrect_value(incorrect_type='computation_miss')
        if mc.get_value() != incorrect_value:
            ans['feedback'] = r'計算が間違っているようです。丁寧に確認しましょう。'
            ans['data'] = incorrect_value
            answers.append(dict(ans))
        incorrect_value = mc.get_incorrect_value(incorrect_type='mul_instead_dot')
        if mc.get_value() != incorrect_value:
            ans['feedback'] = r'行列の計算の規則を誤って覚えています。確認しましょう。'
            ans['data'] = incorrect_value
            answers.append(dict(ans))
        answers = common.answer_union(answers)
        while len(answers) < size:
            incorrect_value = mc.get_incorrect_value(incorrect_type='mixed')
            if mc.get_value() != incorrect_value:
                ans['feedback'] = r'様々な点で間違っています。計算方法を再確認しましょう。'
                ans['data'] = incorrect_value
                answers.append(dict(ans))            
            answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        mc = quiz.data
        _text = r'次の計算をしてください。<br />'
        _text += mc.str_expression(is_latex_closure=True,is_symbolic=False)
        return _text
    def answer_text(self, ans):
        return r'\( ' +  sympy.latex(ans['data'], mat_delim='', mat_str='pmatrix') + r' \)'


# In[118]:


if __name__ == "__main__":
    q = matrix_scalar_computation_Rmn()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[119]:


if __name__ == "__main__":
    pass
    #qz.save('matrix_scalar_computation_Rmn.xml')


# ## matrix-matrix addition in R^(mxn)

# In[114]:


class matrix_matrix_addition_Rmn(core.Question):
    name = '行列同士の和(m行n列の整数行列)'
    def __init__(self, dmin=3, dmax=4, emin=-3, emax=3):
        # 生成する行列のサイズの範囲
        self.dim_min = dmin
        self.dim_max = dmax
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='行列同士の和(mxn)', quiz_number=_quiz_number)
        mc = linear_algebra_next_generation.MatrixComputations()
        mc.set_dimension_range(self.dim_min,self.dim_max)
        mc.set_element_range(self.elem_min,self.elem_max)
        mc.generate("m_m_mn")
        quiz.quiz_identifier = hash(mc)
        # 正答の選択肢の生成
        quiz.data = mc
        ans = { 'fraction': 100, 'data': mc.get_value() }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        mc = quiz.data
        incorrect_value = mc.get_incorrect_value(incorrect_type='forgot_rest')
        if mc.get_value() != incorrect_value:
            ans['feedback'] = r'計算が途中で終わっているようです。丁寧に確認しましょう。'
            ans['data'] = incorrect_value
            answers.append(dict(ans))
        incorrect_value = mc.get_incorrect_value(incorrect_type='forgot_next_cols')
        if mc.get_value() != incorrect_value:
            ans['feedback'] = r'計算が行列の途中で終わっているようです。丁寧に確認しましょう。'
            ans['data'] = incorrect_value
            answers.append(dict(ans))
        incorrect_value = mc.get_incorrect_value(incorrect_type='forgot_next_rows')
        if mc.get_value() != incorrect_value:
            ans['feedback'] = r'計算が行列の途中で終わっているようです。丁寧に確認しましょう。'
            ans['data'] = incorrect_value
            answers.append(dict(ans))
        incorrect_value = mc.get_incorrect_value(incorrect_type='computation_miss')
        if mc.get_value() != incorrect_value:
            ans['feedback'] = r'計算が間違っているようです。丁寧に確認しましょう。'
            ans['data'] = incorrect_value
            answers.append(dict(ans))
        incorrect_value = mc.get_incorrect_value(incorrect_type='mul_instead_dot')
        if mc.get_value() != incorrect_value:
            ans['feedback'] = r'行列の計算の規則を誤って覚えています。確認しましょう。'
            ans['data'] = incorrect_value
            answers.append(dict(ans))
        answers = common.answer_union(answers)
        while len(answers) < size:
            incorrect_value = mc.get_incorrect_value(incorrect_type='mixed')
            if mc.get_value() != incorrect_value:
                ans['feedback'] = r'様々な点で間違っています。計算方法を再確認しましょう。'
                ans['data'] = incorrect_value
                answers.append(dict(ans))            
            answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        mc = quiz.data
        _text = r'次の計算をしてください。<br />'
        _text += mc.str_expression(is_latex_closure=True,is_symbolic=False)
        return _text
    def answer_text(self, ans):
        return r'\( ' +  sympy.latex(ans['data'], mat_delim='', mat_str='pmatrix') + r' \)'


# In[115]:


if __name__ == "__main__":
    q = matrix_matrix_addition_Rmn()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[116]:


if __name__ == "__main__":
    pass
    #qz.save('matrix_matrix_addition_Rmn.xml')


# ## matrix-vector multiplication in R^(mxn)

# In[111]:


class matrix_vector_multiplication_Rmn(core.Question):
    name = '行列とベクトルの積(m行n列の整数行列)'
    def __init__(self, dmin=3, dmax=4, emin=-3, emax=3):
        # 生成する行列のサイズの範囲
        self.dim_min = dmin
        self.dim_max = dmax
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='行列とベクトルの積(mxn)', quiz_number=_quiz_number)
        mc = linear_algebra_next_generation.MatrixComputations()
        mc.set_dimension_range(self.dim_min,self.dim_max)
        mc.set_element_range(self.elem_min,self.elem_max)
        mc.generate("mv_mn")
        quiz.quiz_identifier = hash(mc)
        # 正答の選択肢の生成
        quiz.data = mc
        ans = { 'fraction': 100, 'data': mc.get_value() }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        mc = quiz.data
        incorrect_value = mc.get_incorrect_value(incorrect_type='forgot_rest')
        if mc.get_value() != incorrect_value:
            ans['feedback'] = r'計算が途中で終わっているようです。丁寧に確認しましょう。'
            ans['data'] = incorrect_value
            answers.append(dict(ans))
        incorrect_value = mc.get_incorrect_value(incorrect_type='forgot_next_cols')
        if mc.get_value() != incorrect_value:
            ans['feedback'] = r'計算が行列の途中で終わっているようです。丁寧に確認しましょう。'
            ans['data'] = incorrect_value
            answers.append(dict(ans))
        incorrect_value = mc.get_incorrect_value(incorrect_type='forgot_next_rows')
        if mc.get_value() != incorrect_value:
            ans['feedback'] = r'計算が行列の途中で終わっているようです。丁寧に確認しましょう。'
            ans['data'] = incorrect_value
            answers.append(dict(ans))
        incorrect_value = mc.get_incorrect_value(incorrect_type='computation_miss')
        if mc.get_value() != incorrect_value:
            ans['feedback'] = r'計算が間違っているようです。丁寧に確認しましょう。'
            ans['data'] = incorrect_value
            answers.append(dict(ans))
        incorrect_value = mc.get_incorrect_value(incorrect_type='mul_instead_dot')
        if mc.get_value() != incorrect_value:
            ans['feedback'] = r'行列の計算の規則を誤って覚えています。確認しましょう。'
            ans['data'] = incorrect_value
            answers.append(dict(ans))
        answers = common.answer_union(answers)
        while len(answers) < size:
            incorrect_value = mc.get_incorrect_value(incorrect_type='mixed')
            if mc.get_value() != incorrect_value:
                ans['feedback'] = r'様々な点で間違っています。計算方法を再確認しましょう。'
                ans['data'] = incorrect_value
                answers.append(dict(ans))            
            answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        mc = quiz.data
        _text = r'次の計算をしてください。<br />'
        _text += mc.str_expression(is_latex_closure=True,is_symbolic=False)
        return _text
    def answer_text(self, ans):
        return r'\( ' +  sympy.latex(ans['data'], mat_delim='', mat_str='pmatrix') + r' \)'


# In[112]:


if __name__ == "__main__":
    q = matrix_vector_multiplication_Rmn()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[113]:


if __name__ == "__main__":
    pass
    #qz.save('matrix_vector_multiplication_Rmn.xml')


# ## matrix-matrix multiplication in R^(mxn)

# In[120]:


class matrix_matrix_multiplication_Rmn(core.Question):
    name = '行列同士の積(m行n列の整数行列)'
    def __init__(self, dmin=2, dmax=4, emin=-3, emax=3):
        # 生成する行列のサイズの範囲
        self.dim_min = dmin
        self.dim_max = dmax
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='行列同士の積(mxn)', quiz_number=_quiz_number)
        mc = linear_algebra_next_generation.MatrixComputations()
        mc.set_dimension_range(self.dim_min,self.dim_max)
        mc.set_element_range(self.elem_min,self.elem_max)
        mc.generate("mm_mn")
        quiz.quiz_identifier = hash(mc)
        # 正答の選択肢の生成
        quiz.data = mc
        ans = { 'fraction': 100, 'data': mc.get_value() }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        mc = quiz.data
        incorrect_value = mc.get_incorrect_value(incorrect_type='forgot_rest')
        if mc.get_value() != incorrect_value:
            ans['feedback'] = r'計算が途中で終わっているようです。丁寧に確認しましょう。'
            ans['data'] = incorrect_value
            answers.append(dict(ans))
        incorrect_value = mc.get_incorrect_value(incorrect_type='forgot_next_cols')
        if mc.get_value() != incorrect_value:
            ans['feedback'] = r'計算が行列の途中で終わっているようです。丁寧に確認しましょう。'
            ans['data'] = incorrect_value
            answers.append(dict(ans))
        incorrect_value = mc.get_incorrect_value(incorrect_type='forgot_next_rows')
        if mc.get_value() != incorrect_value:
            ans['feedback'] = r'計算が行列の途中で終わっているようです。丁寧に確認しましょう。'
            ans['data'] = incorrect_value
            answers.append(dict(ans))
        incorrect_value = mc.get_incorrect_value(incorrect_type='computation_miss')
        if mc.get_value() != incorrect_value:
            ans['feedback'] = r'計算が間違っているようです。丁寧に確認しましょう。'
            ans['data'] = incorrect_value
            answers.append(dict(ans))
        incorrect_value = mc.get_incorrect_value(incorrect_type='mul_instead_dot')
        if mc.get_value() != incorrect_value:
            ans['feedback'] = r'行列の計算の規則を誤って覚えています。確認しましょう。'
            ans['data'] = incorrect_value
            answers.append(dict(ans))
        answers = common.answer_union(answers)
        while len(answers) < size:
            incorrect_value = mc.get_incorrect_value(incorrect_type='mixed')
            if mc.get_value() != incorrect_value:
                ans['feedback'] = r'様々な点で間違っています。計算方法を再確認しましょう。'
                ans['data'] = incorrect_value
                answers.append(dict(ans))            
            answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        mc = quiz.data
        _text = r'次の計算をしてください。<br />'
        _text += mc.str_expression(is_latex_closure=True,is_symbolic=False)
        return _text
    def answer_text(self, ans):
        return r'\( ' +  sympy.latex(ans['data'], mat_delim='', mat_str='pmatrix') + r' \)'


# In[121]:


if __name__ == "__main__":
    q = matrix_matrix_multiplication_Rmn()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[122]:


if __name__ == "__main__":
    pass
    #qz.save('matrix_matrix_multiplication_Rmn.xml')


# ## matrix-expression computation in R^(mxn)

# In[123]:


class matrix_expression_computation_Rmn(core.Question):
    name = '行列の式の計算(m行n列の整数行列)'
    def __init__(self, dmin=2, dmax=4, emin=-3, emax=3):
        # 生成する行列のサイズの範囲
        self.dim_min = dmin
        self.dim_max = dmax
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='行列の式の計算(mxn)', quiz_number=_quiz_number)
        mc = linear_algebra_next_generation.MatrixComputations()
        mc.set_dimension_range(self.dim_min,self.dim_max)
        mc.set_element_range(self.elem_min,self.elem_max)
        mc.generate("msvm_mn")
        quiz.quiz_identifier = hash(mc)
        # 正答の選択肢の生成
        quiz.data = mc
        ans = { 'fraction': 100, 'data': mc.get_value() }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        mc = quiz.data
        incorrect_value = mc.get_incorrect_value(incorrect_type='forgot_rest')
        if mc.get_value() != incorrect_value:
            ans['feedback'] = r'計算が途中で終わっているようです。丁寧に確認しましょう。'
            ans['data'] = incorrect_value
            answers.append(dict(ans))
        incorrect_value = mc.get_incorrect_value(incorrect_type='forgot_next_cols')
        if mc.get_value() != incorrect_value:
            ans['feedback'] = r'計算が行列の途中で終わっているようです。丁寧に確認しましょう。'
            ans['data'] = incorrect_value
            answers.append(dict(ans))
        incorrect_value = mc.get_incorrect_value(incorrect_type='forgot_next_rows')
        if mc.get_value() != incorrect_value:
            ans['feedback'] = r'計算が行列の途中で終わっているようです。丁寧に確認しましょう。'
            ans['data'] = incorrect_value
            answers.append(dict(ans))
        incorrect_value = mc.get_incorrect_value(incorrect_type='computation_miss')
        if mc.get_value() != incorrect_value:
            ans['feedback'] = r'計算が間違っているようです。丁寧に確認しましょう。'
            ans['data'] = incorrect_value
            answers.append(dict(ans))
        incorrect_value = mc.get_incorrect_value(incorrect_type='mul_instead_dot')
        if mc.get_value() != incorrect_value:
            ans['feedback'] = r'行列の計算の規則を誤って覚えています。確認しましょう。'
            ans['data'] = incorrect_value
            answers.append(dict(ans))
        answers = common.answer_union(answers)
        while len(answers) < size:
            incorrect_value = mc.get_incorrect_value(incorrect_type='mixed')
            if mc.get_value() != incorrect_value:
                ans['feedback'] = r'様々な点で間違っています。計算方法を再確認しましょう。'
                ans['data'] = incorrect_value
                answers.append(dict(ans))            
            answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        mc = quiz.data
        _text = r'次の行列等に対して，' + mc.str_expression(is_latex_closure=True,is_symbolic=True) + r'を計算してください。<br />'
        _text += mc.str_definition(is_latex_closure=True)
        return _text
    def answer_text(self, ans):
        return r'\( ' +  sympy.latex(ans['data'], mat_delim='', mat_str='pmatrix') + r' \)'


# In[125]:


if __name__ == "__main__":
    q = matrix_expression_computation_Rmn()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[126]:


if __name__ == "__main__":
    pass
    #qz.save('matrix_expression_computation_Rmn.xml')


# ## conversion between polynomial and matrix equations

# In[44]:


class conversion_poly_matrix_eqs(core.Question):
    name = '線形方程式の変換（多項式と行列間の変換）'
    def __init__(self, dmin=2, dmax=4, emin=-3, emax=3, zratio=0.25):
        # 未知数の個数の範囲
        self.dim_min = dmin
        self.dim_max = dmax
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 係数がゼロとなる確率
        self.zero_ratio = zratio
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='線形方程式の表現の変換', quiz_number=_quiz_number)
        if random.random() < 0.5:
            _is_poly2mat = True
        else:
            _is_poly2mat = False
        _num_of_variables = random.randint(self.dim_min,self.dim_max)
        _num_of_equations = random.randint(max(1, self.dim_min-1), self.dim_max+1)
        _matA = [[linear_algebra_next_generation.sometimes_zero_randint(self.elem_min,self.elem_max,self.zero_ratio) for i in range(_num_of_variables)] for j in range(_num_of_equations)] 
        while self._is_zero_row_vector(_matA):
            _matA = [[linear_algebra_next_generation.sometimes_zero_randint(self.elem_min,self.elem_max,self.zero_ratio) for i in range(_num_of_variables)] for j in range(_num_of_equations)] 
        _vecB = [linear_algebra_next_generation.sometimes_zero_randint(self.elem_min,self.elem_max,self.zero_ratio) for j in range(_num_of_equations)] 
        quiz.quiz_identifier = hash(str(_matA)+str(_vecB)+str(_is_poly2mat))
        # 正答の選択肢の生成
        quiz.data = [_is_poly2mat, _num_of_variables, _num_of_equations, _matA, _vecB]
        ans = { 'fraction': 100, 'data': [not(_is_poly2mat), _num_of_variables, _num_of_equations, _matA, _vecB] }
        quiz.answers.append(ans)
        return quiz        
    @staticmethod
    def _is_zero_row_vector(matA):
        _is_zero_vector = False
        for j in range(len(matA)):
            _sum = 0
            for i in range(len(matA[0])):
                _sum += sympy.Abs(matA[j][i])
            if _sum == 0:
                _is_zero_vector = True
        return _is_zero_vector
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_is_poly2mat, _num_of_variables, _num_of_equations, _matA, _vecB] = quiz.data
        incorrect_nv = _num_of_equations
        incorrect_ne = _num_of_variables
        incorrect_mA = sympy.Matrix(_matA).transpose().tolist()
        if len(_vecB) < incorrect_ne:
            incorrect_vB = _vecB + [0 for i in range(incorrect_ne - len(_vecB))]
        elif len(_vecB) > incorrect_ne:
            incorrect_vB = _vecB[:incorrect_ne]
        else:
            incorrect_vB = _vecB
        if sympy.Matrix(incorrect_mA) != sympy.Matrix(_matA) and not(self._is_zero_row_vector(incorrect_mA)):
            ans['feedback'] = r'行列とベクトルの積の演算を用いて，相互に変換してください。'
            ans['data'] = [not(_is_poly2mat), incorrect_nv, incorrect_ne, incorrect_mA, incorrect_vB]
            answers.append(dict(ans))
        incorrect_mA = [[e for e in reversed(vec)] for vec in incorrect_mA]
        if sympy.Matrix(incorrect_mA) != sympy.Matrix(_matA) and not(self._is_zero_row_vector(incorrect_mA)):
            ans['feedback'] = r'行列とベクトルの積の演算を用いて，相互に変換してください。色々と間違っています。'
            ans['data'] = [not(_is_poly2mat), incorrect_nv, incorrect_ne, incorrect_mA, incorrect_vB]
            answers.append(dict(ans))
        incorrect_mA = [[e for e in reversed(vec)] for vec in _matA]
        if sympy.Matrix(incorrect_mA) != sympy.Matrix(_matA):
            ans['feedback'] = r'行列とベクトルの積の演算を用いて，相互に変換してください。なぜか未知数の順序が入れ替わっています。'
            ans['data'] = [not(_is_poly2mat), _num_of_variables, _num_of_equations, incorrect_mA, _vecB]
            answers.append(dict(ans))
        incorrect_mA = []
        for vec in _matA:
            _incorrect_row = []
            _incorrect_zero = []
            for e in vec:
                if e != 0:
                    _incorrect_row.append(e)
                else:
                    _incorrect_zero.append(0)
            incorrect_mA.append(_incorrect_row + _incorrect_zero)
        if sympy.Matrix(incorrect_mA) != sympy.Matrix(_matA):
            ans['feedback'] = r'行列とベクトルの積の演算を用いて，相互に変換してください。未知数を飛ばさないように気をつけてください。'
            ans['data'] = [not(_is_poly2mat), _num_of_variables, _num_of_equations, incorrect_mA, _vecB]
            answers.append(dict(ans))
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_is_poly2mat, _num_of_variables, _num_of_equations, _matA, _vecB] = quiz.data
        _vars = []
        for i in range(1,_num_of_variables+1):
            _vars.append(sympy.symbols('x'+str(i)))
        _text = r'次の線形方程式と同じ方程式を選択してください。<br />'
        if _is_poly2mat:
            _text += r'\( \left\{\begin{array}{c}' + "\n"
            for j in range(_num_of_equations):
                _eq = 0
                for i in range(_num_of_variables):
                    _eq += _matA[j][i]*_vars[i]
                _text += sympy.latex(sympy.Eq(_eq,_vecB[j])) + r'\\' + "\n"
            _text += r'\end{array}\right. \)'
        else:
            _text += r'\( ' +  sympy.latex(sympy.Matrix(_matA), mat_delim='', mat_str='pmatrix') 
            _text += sympy.latex(sympy.Matrix([[v] for v in _vars]), mat_delim='', mat_str='pmatrix') 
            _text += r'='
            _text += sympy.latex(sympy.Matrix([[e] for e in _vecB]), mat_delim='', mat_str='pmatrix') 
            _text += r' \)'
        return _text
    def answer_text(self, ans):
        [_is_poly2mat, _num_of_variables, _num_of_equations, _matA, _vecB] = ans['data']
        _vars = []
        for i in range(1,_num_of_variables+1):
            _vars.append(sympy.symbols('x'+str(i)))
        if _is_poly2mat:
            _text = r'\( \left\{\begin{array}{c}' + "\n"
            for j in range(_num_of_equations):
                _eq = 0
                for i in range(_num_of_variables):
                    _eq += _matA[j][i]*_vars[i]
                _text += sympy.latex(sympy.Eq(_eq,_vecB[j])) + r'\\' + "\n"
            _text += r'\end{array}\right. \)'
        else:
            _text = r'\( ' +  sympy.latex(sympy.Matrix(_matA), mat_delim='', mat_str='pmatrix') 
            _text += sympy.latex(sympy.Matrix([[v] for v in _vars]), mat_delim='', mat_str='pmatrix') 
            _text += r'='
            _text += sympy.latex(sympy.Matrix([[e] for e in _vecB]), mat_delim='', mat_str='pmatrix') 
            _text += r' \)'
        return _text


# In[45]:


if __name__ == "__main__":
    q = conversion_poly_matrix_eqs()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[46]:


if __name__ == "__main__":
    pass
    #qz.save('conversion_poly_matrix_eqs.xml')


# ## conversion between equation and augmented matrix

# In[47]:


class conversion_eqn_augmented_mat(core.Question):
    name = '線形方程式の変換（方程式と拡大係数行列の変換）'
    def __init__(self, dmin=2, dmax=4, emin=-3, emax=3, zratio=0.25):
        # 未知数の個数の範囲
        self.dim_min = dmin
        self.dim_max = dmax
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 係数がゼロとなる確率
        self.zero_ratio = zratio
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='線形方程式と拡大係数行列間の変換', quiz_number=_quiz_number)
        if random.random() < 0.5:
            _is_poly2mat = True
        else:
            _is_poly2mat = False
        _num_of_variables = random.randint(self.dim_min,self.dim_max)
        _num_of_equations = random.randint(max(1, self.dim_min-1), self.dim_max+1)
        _matA = [[linear_algebra_next_generation.sometimes_zero_randint(self.elem_min,self.elem_max,self.zero_ratio) for i in range(_num_of_variables)] for j in range(_num_of_equations)] 
        while self._is_zero_row_vector(_matA):
            _matA = [[linear_algebra_next_generation.sometimes_zero_randint(self.elem_min,self.elem_max,self.zero_ratio) for i in range(_num_of_variables)] for j in range(_num_of_equations)] 
        _vecB = [linear_algebra_next_generation.sometimes_zero_randint(self.elem_min,self.elem_max,self.zero_ratio) for j in range(_num_of_equations)] 
        quiz.quiz_identifier = hash(str(_matA)+str(_vecB)+str(_is_poly2mat))
        # 正答の選択肢の生成
        quiz.data = [_is_poly2mat, _num_of_variables, _num_of_equations, _matA, _vecB]
        ans = { 'fraction': 100, 'data': [not(_is_poly2mat), _num_of_variables, _num_of_equations, _matA, _vecB] }
        quiz.answers.append(ans)
        return quiz        
    @staticmethod
    def _is_zero_row_vector(matA):
        _is_zero_vector = False
        for j in range(len(matA)):
            _sum = 0
            for i in range(len(matA[0])):
                _sum += sympy.Abs(matA[j][i])
            if _sum == 0:
                _is_zero_vector = True
        return _is_zero_vector
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_is_poly2mat, _num_of_variables, _num_of_equations, _matA, _vecB] = quiz.data
        incorrect_nv = _num_of_equations
        incorrect_ne = _num_of_variables
        incorrect_mA = sympy.Matrix(_matA).transpose().tolist()
        if len(_vecB) < incorrect_ne:
            incorrect_vB = _vecB + [0 for i in range(incorrect_ne - len(_vecB))]
        elif len(_vecB) > incorrect_ne:
            incorrect_vB = _vecB[:incorrect_ne]
        else:
            incorrect_vB = _vecB
        if sympy.Matrix(incorrect_mA) != sympy.Matrix(_matA) and not(self._is_zero_row_vector(incorrect_mA)):
            ans['feedback'] = r'行列とベクトルの積の演算を用いて，相互に変換してください。'
            ans['data'] = [not(_is_poly2mat), incorrect_nv, incorrect_ne, incorrect_mA, incorrect_vB]
            answers.append(dict(ans))
        incorrect_mA = [[e for e in reversed(vec)] for vec in incorrect_mA]
        if sympy.Matrix(incorrect_mA) != sympy.Matrix(_matA) and not(self._is_zero_row_vector(incorrect_mA)):
            ans['feedback'] = r'行列とベクトルの積の演算を用いて，相互に変換してください。色々と間違っています。'
            ans['data'] = [not(_is_poly2mat), incorrect_nv, incorrect_ne, incorrect_mA, incorrect_vB]
            answers.append(dict(ans))
        incorrect_mA = [[e for e in reversed(vec)] for vec in _matA]
        if sympy.Matrix(incorrect_mA) != sympy.Matrix(_matA):
            ans['feedback'] = r'行列とベクトルの積の演算を用いて，相互に変換してください。なぜか未知数の順序が入れ替わっています。'
            ans['data'] = [not(_is_poly2mat), _num_of_variables, _num_of_equations, incorrect_mA, _vecB]
            answers.append(dict(ans))
        incorrect_mA = []
        for vec in _matA:
            _incorrect_row = []
            _incorrect_zero = []
            for e in vec:
                if e != 0:
                    _incorrect_row.append(e)
                else:
                    _incorrect_zero.append(0)
            incorrect_mA.append(_incorrect_row + _incorrect_zero)
        if sympy.Matrix(incorrect_mA) != sympy.Matrix(_matA):
            ans['feedback'] = r'行列とベクトルの積の演算を用いて，相互に変換してください。未知数を飛ばさないように気をつけてください。'
            ans['data'] = [not(_is_poly2mat), _num_of_variables, _num_of_equations, incorrect_mA, _vecB]
            answers.append(dict(ans))
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_is_poly2mat, _num_of_variables, _num_of_equations, _matA, _vecB] = quiz.data
        _vars = []
        for i in range(1,_num_of_variables+1):
            _vars.append(sympy.symbols('x'+str(i)))
        if _is_poly2mat:
            _text = r'次の線形方程式の拡大係数行列を選択してください（未知数は\( x_1,x_2,\ldots \)）。<br />'
        else:
            _text = r'次の行列が拡大係数行列となる線形方程式を選択してください（未知数は\( x_1,x_2,\ldots \)）。<br />'
        if _is_poly2mat:
            _text += r'\( \left\{\begin{array}{c}' + "\n"
            for j in range(_num_of_equations):
                _eq = 0
                for i in range(_num_of_variables):
                    _eq += _matA[j][i]*_vars[i]
                _text += sympy.latex(sympy.Eq(_eq,_vecB[j])) + r'\\' + "\n"
            _text += r'\end{array}\right. \)'
        else:
            _text += r'\( ' +  sympy.latex(sympy.Matrix([_matA[i] + [_vecB[i]] for i in range(_num_of_equations)]), mat_delim='', mat_str='pmatrix') + r' \)'
        return _text
    def answer_text(self, ans):
        [_is_poly2mat, _num_of_variables, _num_of_equations, _matA, _vecB] = ans['data']
        _vars = []
        for i in range(1,_num_of_variables+1):
            _vars.append(sympy.symbols('x'+str(i)))
        if _is_poly2mat:
            _text = r'\( \left\{\begin{array}{c}' + "\n"
            for j in range(_num_of_equations):
                _eq = 0
                for i in range(_num_of_variables):
                    _eq += _matA[j][i]*_vars[i]
                _text += sympy.latex(sympy.Eq(_eq,_vecB[j])) + r'\\' + "\n"
            _text += r'\end{array}\right. \)'
        else:
            _text = r'\( ' +  sympy.latex(sympy.Matrix([_matA[i] + [_vecB[i]] for i in range(_num_of_equations)]), mat_delim='', mat_str='pmatrix') + r' \)'
        return _text


# In[48]:


if __name__ == "__main__":
    q = conversion_eqn_augmented_mat()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[49]:


if __name__ == "__main__":
    pass
    #qz.save('conversion_eqn_augmented_mat.xml')


# ## result by elementary row operation

# In[11]:


class result_by_elementary_row_operation(core.Question):
    name = '行の基本変形の適用結果（指定された操作の結果を選ぶ）'
    op_types = ['n->kn', 'n<->m', 'n->n+km']
    def __init__(self, dmin=2, dmax=4, emin=-3, emax=3):
        # 行列のサイズの範囲
        self.dim_min = dmin
        self.dim_max = dmax
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='行の基本変形の適用結果', quiz_number=_quiz_number)
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
        # 正答の選択肢の生成
        quiz.data = [_rows, _cols, _matA, _op_type, _k, _row, _row2, _matB]
        ans = { 'fraction': 100, 'data': _matB }
        quiz.answers.append(ans)
        return quiz      
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
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
                        ans['feedback'] = r'指定された行の基本変形を単純に適用してください。'
                        ans['data'] = incorrectB
                        answers.append(dict(ans))
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_rows, _cols, _matA, _op_type, _k, _row, _row2, _matB] = quiz.data
        _text = r'次の行列に行の基本変形「'
        if _op_type == 'n->kn':
            _text += r'第' + str(_row+1) + r'行を，' + str(_k) + r'倍する'
        elif _op_type == 'n<->m':
            _text += r'第' + str(_row+1) + r'行と第' + str(_row2+1) + r'行を交換する'
        else: # 'n->n+km'
            _text += r'第' + str(_row+1) + r'行に，第' + str(_row2+1) + r'行の' + str(_k) + r'倍を加える'       
        _text += r'」を行った結果の行列を選んでください。<br />'
        _text += r'\( ' +  sympy.latex(sympy.Matrix(_matA), mat_delim='', mat_str='pmatrix') + r' \)'
        return _text
    def answer_text(self, ans):
        return r'\( ' +  sympy.latex(sympy.Matrix(ans['data']), mat_delim='', mat_str='pmatrix') + r' \)'


# In[12]:


if __name__ == "__main__":
    q = result_by_elementary_row_operation()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[13]:


if __name__ == "__main__":
    pass
    #qz.save('result_by_elementary_row_operation.xml')


# ## elementary row operation for result

# In[5]:


class elementary_row_operation_for_result(core.Question):
    name = '適切な行の基本変形の選択（指定された結果を得るための変形を選ぶ）'
    op_types = ['n->kn', 'n<->m', 'n->n+km']
    def __init__(self, dmin=2, dmax=4, emin=-3, emax=3):
        # 行列のサイズの範囲
        self.dim_min = dmin
        self.dim_max = dmax
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='適切な行の基本変形の選択', quiz_number=_quiz_number)
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
        # 正答の選択肢の生成
        quiz.data = [_rows, _cols, _matA, _op_type, _k, _row, _row2, _matB]
        ans = { 'fraction': 100, 'data': [_op_type, _k, _row, _row2] }
        quiz.answers.append(ans)
        return quiz      
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
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
                        ans['feedback'] = r'実際に選択肢の行の基本変形を適用することで簡単に確認できます。'
                        ans['data'] = [t, _k, i, j]
                        answers.append(dict(ans))
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_rows, _cols, _matA, _op_type, _k, _row, _row2, _matB] = quiz.data
        _text = r'次の行列\( A \)から行列\( B \)が得るために必要な行の基本変形を選択してください。<br />'
        _text += r'\( A=' + sympy.latex(sympy.Matrix(_matA), mat_delim='', mat_str='pmatrix')
        _text += r',\;B=' + sympy.latex(sympy.Matrix(_matB), mat_delim='', mat_str='pmatrix') + r' \)'
        return _text
    def answer_text(self, ans):
        [_op_type, _k, _row, _row2] = ans['data']
        if _op_type == 'n->kn':
            return r'第' + str(_row+1) + r'行を，' + str(_k) + r'倍する'
        elif _op_type == 'n<->m':
            return r'第' + str(_row+1) + r'行と第' + str(_row2+1) + r'行を交換する'
        else: # 'n->n+km'
            return r'第' + str(_row+1) + r'行に，第' + str(_row2+1) + r'行の' + str(_k) + r'倍を加える' 


# In[7]:


if __name__ == "__main__":
    q = elementary_row_operation_for_result()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[8]:


if __name__ == "__main__":
    pass
    #qz.save('elementary_row_operation_for_result.xml')


# ## reduced row echelon form of full rank matrix of nn1

# In[22]:


class rref_full_rank_matrix_nn1(core.Question):
    name = '簡約な行列を求める（(n,n+1)の階数nの行列）'
    op_types = ['n->kn', 'n<->m', 'n->n+km']
    def __init__(self, dmin=2, dmax=4, emin=-3, emax=3, smin=2, smax=3):
        # 行列のサイズの範囲
        self.dim_min = dmin
        self.dim_max = dmax
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 行の基本変形の回数の範囲
        self.shuffle_min = smin
        self.shuffle_max = smax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='簡約な行列を求める', quiz_number=_quiz_number)
        _n = random.randint(self.dim_min,self.dim_max)
        _rref = [[1 if i == j else 0 for i in range(_n)] for j in range(_n)]
        _rref = [_row + [random.randint(self.elem_min,self.elem_max)] for _row in _rref]
        _s = random.randint(self.shuffle_min,self.shuffle_max)
        while True:
            _matA = sympy.Matrix(_rref)
            _matB = sympy.Matrix(_rref)
            for i in range(_s):
                _op_type = random.choice(self.op_types)
                _k = random.randint(self.elem_min,self.elem_max)
                _row = random.choice(range(_n))
                _row2 = _row
                while _row == _row2:
                    _row2 = random.choice(range(_n))
                _matB = _matB.elementary_row_op(op=_op_type, row=_row, k=_k, row2=_row2)
            if _matB.rank() == _n and _matB != _matA:
                _matA = _matB.tolist()
                break
        quiz.quiz_identifier = hash(str(_matA)+str(_rref))
        # 正答の選択肢の生成
        quiz.data = [_n, _rref, _matA]
        ans = { 'fraction': 100, 'data': _rref }
        quiz.answers.append(ans)
        return quiz      
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=5):
        answers = []
        ans = { 'fraction': 0 }
        [_n, _rref, _matA] = quiz.data
        _rref_matrix = sympy.Matrix(_rref)
        _incorrect_rref = sympy.eye(_n).tolist()
        ans['feedback'] = r'行の基本変形では，行列のサイズは変わりません。'
        ans['data'] = _incorrect_rref
        answers.append(dict(ans))
        _incorrect_rref = [_incorrect_rref[i] +[_matA[i][_n]] for i in range(_n)]
        if _rref_matrix != sympy.Matrix(_incorrect_rref):
            ans['feedback'] = r'左側だけを単位行列に置き換えるのではなく，行の基本変形をきちんと行ってください。'
            ans['data'] = _incorrect_rref
            answers.append(dict(ans))  
        _incorrect_rref = [[_matA[i][_n] if i == j else 0 for i in range(_n)] for j in range(_n)]
        if _rref_matrix != sympy.Matrix(_incorrect_rref):
            ans['feedback'] = r'行の基本変形をきちんと行ってください。行の基本変形では，行列のサイズは変わりません。'
            ans['data'] = _incorrect_rref
            answers.append(dict(ans))  
        _incorrect_rref = _rref_matrix.elementary_row_op(op='n->kn', row=random.choice(range(_n)), k=random.randint(self.elem_min,self.elem_max)).tolist()
        if _rref_matrix != sympy.Matrix(_incorrect_rref):
            ans['feedback'] = r'対応する線形方程式が解かれた状態というのは，左側が単位行列になったときです。'
            ans['data'] = _incorrect_rref
            answers.append(dict(ans))  
        answers = common.answer_union(answers)
        while len(answers) < size:
            _incorrect_rref = sympy.eye(_n).tolist()
            _incorrect_rref = [_incorrect_rref[i] + [_rref[i][_n] + random.choice([-1,0,1])] for i in range(_n)]
            if _rref_matrix != sympy.Matrix(_incorrect_rref):
                ans['feedback'] = r'行の基本変形をきちんと行ってください。計算が間違っています。'
                ans['data'] = _incorrect_rref
                answers.append(dict(ans))  
            answers = common.answer_union(answers)            
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_n, _rref, _matA] = quiz.data
        _text = r'次の行列に行の基本変形を行い，対応する線形方程式が解かれた状態にしたときに得られる行列を選択してください。<br />'
        _text += r'\( ' +  sympy.latex(sympy.Matrix(_matA), mat_delim='', mat_str='pmatrix') + r' \)'
        return _text
    def answer_text(self, ans):
        return r'\( ' +  sympy.latex(sympy.Matrix(ans['data']), mat_delim='', mat_str='pmatrix') + r' \)'


# In[23]:


if __name__ == "__main__":
    q = rref_full_rank_matrix_nn1()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[24]:


if __name__ == "__main__":
    pass
    #qz.save('rref_full_rank_matrix_nn1.xml')


# ## select all pivots

# In[24]:


class select_all_pivots(core.Question):
    name = '主成分を答える（行主成分）'
    def __init__(self, dmin=2, dmax=5, emin=-3, emax=3, zratio=0.5):
        # 行列のサイズの範囲
        self.dim_min = dmin
        self.dim_max = dmax
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 零成分となる確率
        self.zero_ratio = zratio
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='主成分を答える', quiz_number=_quiz_number)
        _rows = random.randint(self.dim_min,self.dim_max)
        _cols = random.randint(self.dim_min,self.dim_max)
        _mat = [[0 if random.random() <= self.zero_ratio else linear_algebra_next_generation.nonzero_randint(self.dim_min,self.dim_max) for j in range(_cols)] for i in range(_rows)]
        quiz.quiz_identifier = hash(str(_mat))
        # 正答の選択肢の生成
        _pivots = []
        for i in range(_rows):
            for j in range(_cols):
                if _mat[i][j] != 0:
                    _pivots = _pivots + [(i,j)]
                    break
        quiz.data = [_rows, _cols, _mat, _pivots]
        ans = { 'fraction': 100, 'data': _pivots }
        quiz.answers.append(ans)
        return quiz      
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=5):
        answers = []
        ans = { 'fraction': 0 }
        [_rows, _cols, _mat, _pivots] = quiz.data
        _pivots_set = set(_pivots)
        all_indices = [[(i,j) for j in range(_cols)] for i in range(_rows)]
        _incorrect_pivots = [random.choice(all_indices[i]) for i in range(_rows)]
        if _pivots_set != set(_incorrect_pivots):
            ans['feedback'] = r'主成分の条件をきちんと確認しましょう。'
            ans['data'] = _incorrect_pivots
            answers.append(dict(ans))
        _incorrect_pivots = [_p if random.random() <= 0.5 else random.choice(all_indices[_p[0]]) for _p in _pivots]
        if _pivots_set != set(_incorrect_pivots):
            ans['feedback'] = r'主成分の条件をきちんと確認しましょう。'
            ans['data'] = _incorrect_pivots
            answers.append(dict(ans))
        _incorrect_pivots = [(_p[1],_p[0]) for _p in _pivots]
        if _pivots_set != set(_incorrect_pivots):
            ans['feedback'] = r'行列の行と列を取り違えていないでしょうか。'
            ans['data'] = _incorrect_pivots
            answers.append(dict(ans))
        if len(_incorrect_pivots) > 1:
            _incorrect_pivots = random.sample(_incorrect_pivots, k=len(_incorrect_pivots)-1)
            if _pivots_set != set(_incorrect_pivots):
                ans['feedback'] = r'行列の行と列を取り違えていないでしょうか。加えて，個数も不足しています。'
                ans['data'] = _incorrect_pivots
                answers.append(dict(ans))
        _incorrect_pivots = [(_p[0]-1,_p[1]-1) for _p in _incorrect_pivots]
        if _pivots_set != set(_incorrect_pivots):
            ans['feedback'] = r'行列の行番号と列番号は\( 1 \)から始まりますし，行列の行と列を取り違えていないでしょうか。'
            ans['data'] = _incorrect_pivots
            answers.append(dict(ans))
        _incorrect_pivots = [(_p[0]-1,_p[1]-1) for _p in _pivots]
        if _pivots_set != set(_incorrect_pivots):
            ans['feedback'] = r'行列の行番号と列番号は\( 1 \)から始まります。'
            ans['data'] = _incorrect_pivots
            answers.append(dict(ans))
        if len(_incorrect_pivots) > 1:
            _incorrect_pivots = random.sample(_incorrect_pivots, k=len(_incorrect_pivots)-1)
            if _pivots_set != set(_incorrect_pivots):
                ans['feedback'] = r'行列の行番号と列番号は\( 1 \)から始まります。'
                ans['data'] = _incorrect_pivots
                answers.append(dict(ans))
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_rows, _cols, _mat, _pivots] = quiz.data
        _text = r'次の行列のすべての主成分として正しいものを選択してください。<br />'
        _text += r'\( ' +  sympy.latex(sympy.Matrix(_mat), mat_delim='', mat_str='pmatrix') + r' \)'
        return _text
    def answer_text(self, ans):
        _text = ''
        for _p in ans['data']:
            _text += r'\( (' + sympy.latex(_p[0]+1) + r',' + sympy.latex(_p[1]+1) + r') \)成分，'
        return _text[:-1]


# In[25]:


if __name__ == "__main__":
    q = select_all_pivots()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[26]:


if __name__ == "__main__":
    pass
    #qz.save('select_all_pivots.xml')


# ## select reduced row echelon form

# In[28]:


class select_reduced_row_echelon_form(core.Question):
    name = '簡約な行列を選択する（行簡約行列の選択）'
    def __init__(self, dmin=3, dmax=5, emin=-3, emax=3, zratio=0.5):
        # 行列のサイズの範囲
        self.dim_min = dmin
        self.dim_max = dmax
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 零成分となる確率
        self.zero_ratio = zratio
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='行簡約行列の選択', quiz_number=_quiz_number)
        _mr = linear_algebra_next_generation.MatrixInReducedRowEchelonForm()
        _mr.set_dimension_range(self.dim_min,self.dim_max)
        _mr.set_element_range(self.dim_min,self.dim_max)
        _mr.set_zero_vector_ratio(self.zero_ratio)
        _mr.generate()
        quiz.quiz_identifier = hash(_mr)
        # 正答の選択肢の生成
        quiz.data = [_mr]
        ans = { 'fraction': 100, 'data': _mr.get_rref() }
        quiz.answers.append(ans)
        return quiz      
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=5):
        answers = []
        ans = { 'fraction': 0 }
        [_mr] = quiz.data
        _rref = sympy.Matrix(_mr.get_rref())
        for _i in range(2):
            _row1 = random.randint(0, sympy.floor(_rref.rows/2))
            _row2 = random.randint(sympy.floor(_rref.rows/2) + 1, _rref.rows - 1)
            _incorrect_rref = _rref.elementary_row_op(op='n<->m', row=_row1, row2=_row2)
            if _incorrect_rref != _incorrect_rref.rref(pivots=False):
                ans['feedback'] = r'零ベクトルの上下の位置か主成分の上下の位置が条件を満たしていません。'
                ans['data'] = _incorrect_rref
                answers.append(dict(ans))             
        for _i in [-1,2]:
            _row1 = random.randint(0, sympy.floor(_rref.rows/2))
            _incorrect_rref = _rref.elementary_row_op(op='n->kn', row=_row1, k=_i)
            if _incorrect_rref != _incorrect_rref.rref(pivots=False):
                ans['feedback'] = r'主成分は\( 1 \)である必要があります。'
                ans['data'] = _incorrect_rref
                answers.append(dict(ans))         
                _row1 = random.randint(0, sympy.floor(_rref.rows/2))
                _row2 = random.randint(sympy.floor(_rref.rows/2) + 1, _rref.rows - 1)
                _incorrect_rref = _incorrect_rref.elementary_row_op(op='n<->m', row=_row1, row2=_row2)
                if _incorrect_rref != _incorrect_rref.rref(pivots=False):
                    ans['feedback'] = r'簡約な行列の条件を全く理解していない可能性があります。確認してください。'
                    ans['data'] = _incorrect_rref
                    answers.append(dict(ans))        
        for _i in range(2):
            _incorrect_rref = sympy.Matrix(_mr.get_rref())
            _r = random.randint(0, _rref.rows - 1)
            _c = random.randint(0, _rref.cols - 1)
            _incorrect_rref[_r,_c] = linear_algebra_next_generation.nonzero_randint(self.elem_min, self.elem_max)
            if _incorrect_rref != _incorrect_rref.rref(pivots=False):
                ans['feedback'] = r'条件を満たしていない余計な要素があります。よく確認してください。'
                ans['data'] = _incorrect_rref
                answers.append(dict(ans))              
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_mr] = quiz.data
        _text = r'次の行列の中には，簡約な行列がただ1つ含まれています。その行列を選択してください。'
        return _text
    def answer_text(self, ans):
        return r'\( ' + sympy.latex(sympy.Matrix(ans['data']), mat_delim='', mat_str='pmatrix') + r' \)'


# In[29]:


if __name__ == "__main__":
    q = select_reduced_row_echelon_form()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[30]:


if __name__ == "__main__":
    pass
    #qz.save('select_reduced_row_echelon_form.xml')


# ## select row echelon form

# In[16]:


class select_row_echelon_form(core.Question):
    name = '階段行列を選択する（行階段形の行列の選択）'
    def __init__(self, dmin=3, dmax=5, emin=-3, emax=3, zratio=0.5):
        # 行列のサイズの範囲
        self.dim_min = dmin
        self.dim_max = dmax
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 零成分となる確率
        self.zero_ratio = zratio
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='行階段形の行列の選択', quiz_number=_quiz_number)
        _mr = linear_algebra_next_generation.MatrixInRowEchelonForm()
        _mr.set_dimension_range(self.dim_min,self.dim_max)
        _mr.set_element_range(self.dim_min,self.dim_max)
        _mr.set_zero_vector_ratio(self.zero_ratio)
        _mr.generate()
        quiz.quiz_identifier = hash(_mr)
        # 正答の選択肢の生成
        quiz.data = [_mr]
        ans = { 'fraction': 100, 'data': _mr.get_ref() }
        quiz.answers.append(ans)
        return quiz      
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=5):
        answers = []
        ans = { 'fraction': 0 }
        [_mr] = quiz.data
        _rref = sympy.Matrix(_mr.get_ref())
        for _i in range(size):
            _row1 = random.randint(0, sympy.floor(_rref.rows/2))
            _row2 = random.randint(sympy.floor(_rref.rows/2) + 1, _rref.rows - 1)
            _incorrect_rref = _rref.elementary_row_op(op='n<->m', row=_row1, row2=_row2)
            if not _incorrect_rref.is_echelon:
                ans['feedback'] = r'零ベクトルの上下の位置か主成分の上下の位置が条件を満たしていません。'
                ans['data'] = _incorrect_rref
                answers.append(dict(ans))             
        for _i in range(size):
            _incorrect_rref = sympy.Matrix(_mr.get_ref())
            _r = random.randint(0, _rref.rows - 1)
            _c = random.randint(0, _rref.cols - 1)
            _incorrect_rref[_r,_c] = linear_algebra_next_generation.nonzero_randint(self.elem_min, self.elem_max)
            if not _incorrect_rref.is_echelon:
                ans['feedback'] = r'条件を満たしていない余計な要素があります。よく確認してください。'
                ans['data'] = _incorrect_rref
                answers.append(dict(ans))       
                _row1 = random.randint(0, sympy.floor(_rref.rows/2))
                _row2 = random.randint(sympy.floor(_rref.rows/2) + 1, _rref.rows - 1)
                _incorrect_rref = _incorrect_rref.elementary_row_op(op='n<->m', row=_row1, row2=_row2)
                if not _incorrect_rref.is_echelon:
                    ans['feedback'] = r'いくつも条件を満たしていません。よく確認してください。'
                    ans['data'] = _incorrect_rref
                    answers.append(dict(ans))                   
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_mr] = quiz.data
        _text = r'次の行列の中には，階段行列がただ1つ含まれています。その行列を選択してください。'
        return _text
    def answer_text(self, ans):
        return r'\( ' + sympy.latex(sympy.Matrix(ans['data']), mat_delim='', mat_str='pmatrix') + r' \)'


# In[17]:


if __name__ == "__main__":
    q = select_row_echelon_form()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[18]:


if __name__ == "__main__":
    pass
    #qz.save('select_row_echelon_form.xml')


# ## rank of matrix

# In[6]:


class rank_of_matrix(core.Question):
    name = '行列の階数を求める（簡約前の行列の階数）'
    def __init__(self, dmin=3, dmax=4, emin=-2, emax=2, zratio=0.5):
        # 行列のサイズの範囲
        self.dim_min = dmin
        self.dim_max = dmax
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 零成分となる確率
        self.zero_ratio = zratio
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='行列の階数を求める', quiz_number=_quiz_number)
        _mr = linear_algebra_next_generation.MatrixInRowEchelonForm()
        _mr.set_dimension_range(self.dim_min,self.dim_max)
        _mr.set_element_range(self.dim_min,self.dim_max)
        _mr.set_zero_vector_ratio(self.zero_ratio)
        _mr.generate()
        quiz.quiz_identifier = hash(_mr)
        # 正答の選択肢の生成
        quiz.data = [_mr]
        ans = { 'fraction': 100, 'data': _mr.get_rank() }
        quiz.answers.append(ans)
        return quiz      
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=5):
        answers = []
        ans = { 'fraction': 0 }
        [_mr] = quiz.data
        _rref = sympy.Matrix(_mr.get_ref())
        if 0 != _mr.get_rank():
            ans['feedback'] = r'階数が\( 0 \)となるのは，零行列だけです。'
            ans['data'] = 0
            answers.append(dict(ans))             
        for _i in range(1,min(_rref.rows, _rref.cols)+1):
            if _i != _mr.get_rank():
                ans['feedback'] = r'階数を求めるには，階段行列に行の基本変形で変形し，零ベクトルでない行ベクトルを数えてください。'
                ans['data'] = _i
                answers.append(dict(ans))             
        for _i in range(min(_rref.rows, _rref.cols)+1, max(_rref.rows, _rref.cols)+1):
            if _i != _mr.get_rank():
                ans['feedback'] = r'階数は行数と列数のより小さい方よりも大きくはなりません。'
                ans['data'] = _i
                answers.append(dict(ans))             
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_mr] = quiz.data
        _text = r'次の行列の階数を計算し，選択肢の中から正しい階数を選択してください。<br />'
        _text += r'\( ' + sympy.latex(sympy.Matrix(_mr.get_matrix()), mat_delim='', mat_str='pmatrix') + r' \)'
        return _text
    def answer_text(self, ans):
        return r'\( ' + sympy.latex(ans['data']) + r' \)'


# In[7]:


if __name__ == "__main__":
    q = rank_of_matrix()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[8]:


if __name__ == "__main__":
    pass
    #qz.save('rank_of_matrix.xml')


# ## reduced row echelon form

# In[3]:


class reduced_row_echelon_form(core.Question):
    name = '簡約な行列を求める（簡約前の行列から）'
    def __init__(self, dmin=3, dmax=4, emin=-2, emax=2, zratio=0.5):
        # 行列のサイズの範囲
        self.dim_min = dmin
        self.dim_max = dmax
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 零成分となる確率
        self.zero_ratio = zratio
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='簡約な行列を求める', quiz_number=_quiz_number)
        _mr = linear_algebra_next_generation.MatrixInReducedRowEchelonForm()
        _mr.set_dimension_range(self.dim_min,self.dim_max)
        _mr.set_element_range(self.dim_min,self.dim_max)
        _mr.set_zero_vector_ratio(self.zero_ratio)
        _mr.generate()
        quiz.quiz_identifier = hash(_mr)
        # 正答の選択肢の生成
        quiz.data = [_mr]
        ans = { 'fraction': 100, 'data': _mr.get_rref() }
        quiz.answers.append(ans)
        return quiz      
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=5):
        answers = []
        ans = { 'fraction': 0 }
        [_mr] = quiz.data
        _rref = sympy.Matrix(_mr.get_rref())
        _incorrect_rref = _mr.get_row_space_basis()
        if sympy.Matrix(_incorrect_rref) != _rref:
            ans['feedback'] = r'行の基本変形では，行列のサイズは変化しません。'
            ans['data'] = _incorrect_rref
            answers.append(dict(ans))
        for _i in range(size):
            _incorrect_mr = linear_algebra_next_generation.MatrixInReducedRowEchelonForm(_mr)
            _incorrect_mr.generate(is_size_fixed=True)
            _incorrect_rref = _incorrect_mr.get_rref()
            if sympy.Matrix(_incorrect_rref) != _rref:
                ans['feedback'] = r'計算を間違えていないか確認しましょう。'
                ans['data'] = _incorrect_rref
                answers.append(dict(ans))         
                _incorrect_rref = _incorrect_mr.get_row_space_basis()
                if sympy.Matrix(_incorrect_rref) != _rref:
                    ans['feedback'] = r'行の基本変形では，行列のサイズは変化しません。'
                    ans['data'] = _incorrect_rref
                    answers.append(dict(ans))
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_mr] = quiz.data
        _text = r'次の行列を行の基本変形により，簡約な行列にしてください。その行列を選択してください。<br />'
        _text += r'\( ' + sympy.latex(sympy.Matrix(_mr.get_matrix()), mat_delim='', mat_str='pmatrix') + r' \)'
        return _text
    def answer_text(self, ans):
        return r'\( ' + sympy.latex(sympy.Matrix(ans['data']), mat_delim='', mat_str='pmatrix') + r' \)'


# In[6]:


if __name__ == "__main__":
    q = reduced_row_echelon_form()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[7]:


if __name__ == "__main__":
    pass
    #qz.save('reduced_row_echelon_form.xml')


# ## degrees of freedom

# In[6]:


class degrees_of_freedom(core.Question):
    name = '解の自由度を答える（拡大係数行列の簡約な行列から）'
    def __init__(self, dmin=3, dmax=4, emin=-2, emax=2, zratio=0.5, iratio=0.25):
        # 行列のサイズの範囲
        self.dim_min = dmin
        self.dim_max = dmax
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 零成分となる確率
        self.zero_ratio = zratio
        # 解なしの確率
        self.inconsistent_ratio = iratio
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='解の自由度を答える', quiz_number=_quiz_number)
        _le = linear_algebra_next_generation.LinearEquation()
        _le.set_dimension_range(self.dim_min,self.dim_max)
        _le.set_element_range(self.dim_min,self.dim_max)
        _le.set_zero_vector_ratio(self.zero_ratio)
        _le.set_inconsistent_ratio(self.inconsistent_ratio)
        _le.generate()
        quiz.quiz_identifier = hash(_le)
        # 正答の選択肢の生成
        quiz.data = [_le]
        ans = { 'fraction': 100, 'data': _le.get_degrees_of_freedom() }
        quiz.answers.append(ans)
        return quiz      
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=5):
        answers = []
        ans = { 'fraction': 0 }
        [_le] = quiz.data
        _matrix = sympy.Matrix(_le.get_matrix())
        if _le.get_degrees_of_freedom() >= 0:
            ans['feedback'] = r'解がないのは，拡大係数行列の階数と，係数行列の階数が不一致のときです。'
            ans['data'] = -1
            answers.append(dict(ans))
        for i in range(_matrix.cols):
            if i != _le.get_degrees_of_freedom():
                ans['feedback'] = r'未知数の個数から，拡大係数行列の階数を引いたのが，解の自由度です。'
                ans['data'] = i
                answers.append(dict(ans))    
        for i in range(_matrix.cols, max(_matrix.cols,_matrix.rows)+1):
            if i != _le.get_degrees_of_freedom():
                ans['feedback'] = r'解の自由度は，未知数の個数を超えることはありません。'
                ans['data'] = i
                answers.append(dict(ans))            
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_le] = quiz.data
        _text = r'ある線形方程式の拡大係数行列を簡約したところ，次の簡約な行列が得られました。元の線形方程式の解の自由度を選んでください。<br />'
        _text += _le.str_rref(is_latex_closure=True)
        return _text
    def answer_text(self, ans):
        if ans['data'] < 0:
            return r'解はない'
        else:
            return r'\( ' + sympy.latex(ans['data']) + r' \)'


# In[7]:


if __name__ == "__main__":
    q = degrees_of_freedom()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[8]:


if __name__ == "__main__":
    pass
    #qz.save('degrees_of_freedom.xml')


# ## conversion between rref and solution

# In[5]:


class conversion_between_rref_and_solution(core.Question):
    name = '線形方程式の解を構成する（拡大係数行列の簡約な行列から）'
    def __init__(self, dmin=3, dmax=4, emin=-2, emax=2, zratio=0.5, iratio=0.25):
        # 行列のサイズの範囲
        self.dim_min = dmin
        self.dim_max = dmax
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 零成分となる確率
        self.zero_ratio = zratio
        # 解なしの確率
        self.inconsistent_ratio = iratio
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='線形方程式の解を構成する', quiz_number=_quiz_number)
        _le = linear_algebra_next_generation.LinearEquation()
        _le.set_dimension_range(self.dim_min,self.dim_max)
        _le.set_element_range(self.dim_min,self.dim_max)
        _le.set_zero_vector_ratio(self.zero_ratio)
        _le.set_inconsistent_ratio(self.inconsistent_ratio)
        _le.generate()
        quiz.quiz_identifier = hash(_le)
        # 正答の選択肢の生成
        quiz.data = [_le]
        ans = { 'fraction': 100, 'data': _le.str_solution(is_latex_closure=True) }
        quiz.answers.append(ans)
        return quiz      
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass     
    def _str_solution(self, solution_basis):
        _text = r'\( '
        _vars = [sympy.Symbol('x' + str(i+1)) for i in range(len(solution_basis[0]))]
        _text += sympy.latex(sympy.Matrix([_vars]).transpose(), mat_delim='', mat_str='pmatrix')
        _text += r' = '
        for i in range(1,len(solution_basis)):
            _text += sympy.latex(sympy.Matrix([solution_basis[i]]).transpose(), mat_delim='', mat_str='pmatrix')
            _text += r' c_{' + str(i) + r'} + '
        if linear_algebra_next_generation.is_zero_vector(solution_basis[0]) and len(solution_basis) > 1:
            _text = _text[:-2]
        else:
            _text += sympy.latex(sympy.Matrix([solution_basis[0]]).transpose(), mat_delim='', mat_str='pmatrix')
        for i in range(1,len(solution_basis)):
            _text += r',\;c_{' + str(i) + r'}\in\mathbb{R}'
        _text += r' \)'
        return _text            
    def incorrect_answers_generate(self, quiz, size=5):
        answers = []
        ans = { 'fraction': 0 }
        [_le] = quiz.data
        if _le.get_degrees_of_freedom() >= 0:
            ans['feedback'] = r'解がないのは，拡大係数行列の階数と，係数行列の階数が不一致のときです。'
            ans['data'] = r'解なし'
            answers.append(dict(ans))
            for _incorrect_sol in _le.get_fake_solution_basis('basic_fulldrop'):
                ans['feedback'] = r'解の自由度を把握しましょう。自由度の分だけ任意定数が必要です。'
                ans['data'] = self._str_solution(_incorrect_sol)
                answers.append(dict(ans))
            for _incorrect_sol in _le.get_fake_solution_basis('basic_drop'):
                ans['feedback'] = r'解の自由度を把握しましょう。自由度と任意定数の個数が一致していません。'
                ans['data'] = self._str_solution(_incorrect_sol)
                answers.append(dict(ans))
            for _incorrect_sol in _le.get_fake_solution_basis('special_drop'):
                ans['feedback'] = r'定数項に対応する部分を忘れていませんか。'
                ans['data'] = self._str_solution(_incorrect_sol)
                answers.append(dict(ans))
            for _incorrect_sol in _le.get_fake_solution_basis('basic_sign'):
                ans['feedback'] = r'符号にズレが発生しています。簡約結果と解の関係について確認しましょう。'
                ans['data'] = self._str_solution(_incorrect_sol)
                answers.append(dict(ans))
            for _incorrect_sol in _le.get_fake_solution_basis('special_sign'):
                ans['feedback'] = r'符号にズレが発生しています。拡大係数行列の一番右側の列の意味を考えましょう。'
                ans['data'] = self._str_solution(_incorrect_sol)
                answers.append(dict(ans))
            for _incorrect_sol in _le.get_fake_solution_basis('special_random'):
                ans['feedback'] = r'拡大係数行列による表現を，多項式の形に直して，意味を捉えなおしてみましょう。'
                ans['data'] = self._str_solution(_incorrect_sol)
                answers.append(dict(ans))
        else:
            _incorrect_le = linear_algebra_next_generation.LinearEquation(_le)
            _incorrect_le.generate_solution_basis(force_to_generate=True)
            ans['feedback'] = r'拡大係数行列の階数と，係数行列の階数が不一致のときは，解はありません。'
            ans['data'] = _incorrect_le.str_solution(is_latex_closure=True)
            answers.append(dict(ans))
            for _incorrect_sol in _le.get_fake_solution_basis('basic_fulldrop'):
                ans['data'] = self._str_solution(_incorrect_sol)
                answers.append(dict(ans))
            for _incorrect_sol in _le.get_fake_solution_basis('basic_drop'):
                ans['data'] = self._str_solution(_incorrect_sol)
                answers.append(dict(ans))
            for _incorrect_sol in _le.get_fake_solution_basis('special_drop'):
                ans['data'] = self._str_solution(_incorrect_sol)
                answers.append(dict(ans))
            for _incorrect_sol in _le.get_fake_solution_basis('basic_sign'):
                ans['data'] = self._str_solution(_incorrect_sol)
                answers.append(dict(ans))
            for _incorrect_sol in _le.get_fake_solution_basis('special_sign'):
                ans['data'] = self._str_solution(_incorrect_sol)
                answers.append(dict(ans))
            for _incorrect_sol in _le.get_fake_solution_basis('special_random'):
                ans['data'] = self._str_solution(_incorrect_sol)
                answers.append(dict(ans))        
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_le] = quiz.data
        _text = r'ある線形方程式に対応する拡大係数行列を簡約したところ，次の行列になりました。元の線形方程式の解として最も適切と思われるものを選んでください。<br />'
        _text += _le.str_rref(is_latex_closure=True)
        return _text
    def answer_text(self, ans):
        return ans['data']


# In[7]:


if __name__ == "__main__":
    q = conversion_between_rref_and_solution()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[8]:


if __name__ == "__main__":
    pass
    #qz.save('conversion_between_rref_and_solution.xml')


# ## linear equation

# In[6]:


class linear_equation(core.Question):
    name = '線形方程式を解く（一般の問題）'
    def __init__(self, dmin=3, dmax=4, emin=-2, emax=2, zratio=0.5, iratio=0.25):
        # 行列のサイズの範囲
        self.dim_min = dmin
        self.dim_max = dmax
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 零成分となる確率
        self.zero_ratio = zratio
        # 解なしの確率
        self.inconsistent_ratio = iratio
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='線形方程式を解く', quiz_number=_quiz_number)
        _le = linear_algebra_next_generation.LinearEquation()
        _le.set_dimension_range(self.dim_min,self.dim_max)
        _le.set_element_range(self.dim_min,self.dim_max)
        _le.set_zero_vector_ratio(self.zero_ratio)
        _le.set_zero_elements_rest(1)
        _le.set_inconsistent_ratio(self.inconsistent_ratio)
        _le.generate()
        while True:
            _is_ok = True
            for _v in _le.get_matrix():
                if _v[:-1].count(0) >= len(_v) - 2:
                    _is_ok = False
                    break
            if _is_ok:
                break
            _le = linear_algebra_next_generation.LinearEquation()
            _le.set_dimension_range(self.dim_min,self.dim_max)
            _le.set_element_range(self.dim_min,self.dim_max)
            _le.set_zero_vector_ratio(self.zero_ratio)
            _le.set_zero_elements_rest(1)
            _le.set_inconsistent_ratio(self.inconsistent_ratio)
            _le.generate()
        quiz.quiz_identifier = hash(_le)
        # 正答の選択肢の生成
        quiz.data = [_le]
        ans = { 'fraction': 100, 'data': _le.str_solution(is_latex_closure=True) }
        quiz.answers.append(ans)
        return quiz      
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass     
    def _str_solution(self, solution_basis):
        _text = r'\( '
        _vars = [sympy.Symbol('x' + str(i+1)) for i in range(len(solution_basis[0]))]
        _text += sympy.latex(sympy.Matrix([_vars]).transpose(), mat_delim='', mat_str='pmatrix')
        _text += r' = '
        for i in range(1,len(solution_basis)):
            _text += sympy.latex(sympy.Matrix([solution_basis[i]]).transpose(), mat_delim='', mat_str='pmatrix')
            _text += r' c_{' + str(i) + r'} + '
        if linear_algebra_next_generation.is_zero_vector(solution_basis[0]) and len(solution_basis) > 1:
            _text = _text[:-2]
        else:
            _text += sympy.latex(sympy.Matrix([solution_basis[0]]).transpose(), mat_delim='', mat_str='pmatrix')
        for i in range(1,len(solution_basis)):
            _text += r',\;c_{' + str(i) + r'}\in\mathbb{R}'
        _text += r' \)'
        return _text            
    def incorrect_answers_generate(self, quiz, size=5):
        answers = []
        ans = { 'fraction': 0 }
        [_le] = quiz.data
        if _le.get_degrees_of_freedom() >= 0:
            ans['feedback'] = r'解がないのは，拡大係数行列の階数と，係数行列の階数が不一致のときです。'
            ans['data'] = r'解なし'
            answers.append(dict(ans))
            for _incorrect_sol in _le.get_fake_solution_basis('basic_fulldrop'):
                ans['feedback'] = r'解の自由度を把握しましょう。自由度の分だけ任意定数が必要です。'
                ans['data'] = self._str_solution(_incorrect_sol)
                answers.append(dict(ans))
            for _incorrect_sol in _le.get_fake_solution_basis('basic_drop'):
                ans['feedback'] = r'解の自由度を把握しましょう。自由度と任意定数の個数が一致していません。'
                ans['data'] = self._str_solution(_incorrect_sol)
                answers.append(dict(ans))
            for _incorrect_sol in _le.get_fake_solution_basis('special_drop'):
                ans['feedback'] = r'定数項に対応する部分を忘れていませんか。'
                ans['data'] = self._str_solution(_incorrect_sol)
                answers.append(dict(ans))
            for _incorrect_sol in _le.get_fake_solution_basis('basic_sign'):
                ans['feedback'] = r'符号にズレが発生しています。簡約結果と解の関係について確認しましょう。'
                ans['data'] = self._str_solution(_incorrect_sol)
                answers.append(dict(ans))
            for _incorrect_sol in _le.get_fake_solution_basis('special_sign'):
                ans['feedback'] = r'符号にズレが発生しています。拡大係数行列の一番右側の列の意味を考えましょう。'
                ans['data'] = self._str_solution(_incorrect_sol)
                answers.append(dict(ans))
            for _incorrect_sol in _le.get_fake_solution_basis('special_random'):
                ans['feedback'] = r'拡大係数行列による表現を，多項式の形に直して，意味を捉えなおしてみましょう。'
                ans['data'] = self._str_solution(_incorrect_sol)
                answers.append(dict(ans))
        else:
            _incorrect_le = linear_algebra_next_generation.LinearEquation(_le)
            _incorrect_le.generate_solution_basis(force_to_generate=True)
            ans['feedback'] = r'拡大係数行列の階数と，係数行列の階数が不一致のときは，解はありません。'
            ans['data'] = _incorrect_le.str_solution(is_latex_closure=True)
            answers.append(dict(ans))
            for _incorrect_sol in _le.get_fake_solution_basis('basic_fulldrop'):
                ans['data'] = self._str_solution(_incorrect_sol)
                answers.append(dict(ans))
            for _incorrect_sol in _le.get_fake_solution_basis('basic_drop'):
                ans['data'] = self._str_solution(_incorrect_sol)
                answers.append(dict(ans))
            for _incorrect_sol in _le.get_fake_solution_basis('special_drop'):
                ans['data'] = self._str_solution(_incorrect_sol)
                answers.append(dict(ans))
            for _incorrect_sol in _le.get_fake_solution_basis('basic_sign'):
                ans['data'] = self._str_solution(_incorrect_sol)
                answers.append(dict(ans))
            for _incorrect_sol in _le.get_fake_solution_basis('special_sign'):
                ans['data'] = self._str_solution(_incorrect_sol)
                answers.append(dict(ans))
            for _incorrect_sol in _le.get_fake_solution_basis('special_random'):
                ans['data'] = self._str_solution(_incorrect_sol)
                answers.append(dict(ans))        
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_le] = quiz.data
        _text = r'次の線形方程式を解いてください。<br />'
        _text += _le.str_equation(is_latex_closure=True)
        return _text
    def answer_text(self, ans):
        return ans['data']


# In[7]:


if __name__ == "__main__":
    q = linear_equation()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[8]:


if __name__ == "__main__":
    pass
    #qz.save('linear_equation.xml')


# ## dummy

# In[ ]:





# # All the questions

# In[ ]:


questions_str = ['matrix_notation_R22', 'matrix_scalar_computation_R22', 'matrix_matrix_addition_R22', 
                 'matrix_vector_multiplication_R22', 'matrix_matrix_multiplication_R22', 'matrix_expression_computation_R22', 
                 'linear_map_image_of_line_R22', 'matrix_notation_Rmn', 'matrix_scalar_computation_Rmn', 
                 'matrix_matrix_addition_Rmn', 'matrix_vector_multiplication_Rmn', 'matrix_matrix_multiplication_Rmn', 
                 'matrix_expression_computation_Rmn', 'conversion_poly_matrix_eqs', 'conversion_eqn_augmented_mat', 
                 'result_by_elementary_row_operation', 'elementary_row_operation_for_result', 'rref_full_rank_matrix_nn1', 
                 'select_all_pivots', 'select_reduced_row_echelon_form', 'select_row_echelon_form', 'rank_of_matrix', 
                 'reduced_row_echelon_form', 'degrees_of_freedom', 'conversion_between_rref_and_solution', 'linear_equation']
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




