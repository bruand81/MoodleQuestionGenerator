#!/usr/bin/env python
# coding: utf-8

# Moodle XML Question Generator, (C) 2020 Kosaku Nagasaka (Kobe University)

# ## モジュールファイルの生成

# In[1]:


if __name__ == "__main__":
    import subprocess
    subprocess.run(['jupyter', 'nbconvert', '--to', 'python', '_basic_calculus_1.ipynb','--output','basic_calculus_1.py'])


# # Basic Calculus 1 (differencial calculus for univariate function)

# In[3]:


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
    core._force_to_this_lang = 'ja'
else:
    from .. import core
    from . import common


# In[4]:


import sympy
import random
import IPython
import itertools
import copy


# ## minor helpers

# In[5]:


def nonzero_randint(imin, imax):
    if 0 < imin or imax < 0:
        return random.randint(imin, imax)
    else:
        return random.choice(list(set(range(imin,imax+1)).difference(set([0]))))


# In[6]:


def nonone_randpint(imin, imax):
    if 1 < imin:
        return random.randint(imin, imax)
    else:
        return abs(random.choice(list(set(range(imin,imax+1)).difference(set([0,-1,1])))))


# ## composite function

# In[7]:


class composite_function(core.Question):
    name = '合成関数（初等関数）'
    _function_types = ['polynomial', 'polynomial','polynomial', 'rational', 'rational', 'sin', 'cos', 'tan', 'exp', 'log']
    def __init__(self, emin=-2, emax=2):
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
    def _function_generate(self, _var):
        _type = random.choice(self._function_types)
        if _type == 'polynomial':
            return sum([nonzero_randint(self.elem_min, self.elem_max)*_var**i for i in range(0,random.randint(2,3))])
        elif _type == 'rational':
            return sum([nonzero_randint(self.elem_min, self.elem_max)*_var**i for i in range(0,random.randint(1,2))]) / sum([nonzero_randint(self.elem_min, self.elem_max)*_var**i for i in range(0,random.randint(1,2))])
        elif _type == 'sin':
            return nonzero_randint(self.elem_min, self.elem_max)*sympy.sin(nonzero_randint(self.elem_min, self.elem_max)*_var+random.randint(self.elem_min, self.elem_max))
        elif _type == 'cos':
            return nonzero_randint(self.elem_min, self.elem_max)*sympy.cos(nonzero_randint(self.elem_min, self.elem_max)*_var+random.randint(self.elem_min, self.elem_max))
        elif _type == 'tan':
            return nonzero_randint(self.elem_min, self.elem_max)*sympy.tan(nonzero_randint(self.elem_min, self.elem_max)*_var+random.randint(self.elem_min, self.elem_max))
        elif _type == 'exp':
            return nonzero_randint(self.elem_min, self.elem_max)*sympy.exp(nonzero_randint(self.elem_min, self.elem_max)*_var+random.randint(self.elem_min, self.elem_max))
        elif _type == 'log':
            return nonzero_randint(self.elem_min, self.elem_max)*sympy.log(abs(nonzero_randint(self.elem_min, self.elem_max))*_var+abs(random.randint(self.elem_min, self.elem_max)))
        else:
            return 0
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='合成関数', quiz_number=_quiz_number)
        _var = sympy.Symbol('x')
        _f = self._function_generate(_var)
        _g = self._function_generate(_var)
        _is_fog = random.choice([True, False])
        if _is_fog is True:
            _composite = _f.subs(_var, _g)
        else:
            _composite = _g.subs(_var, _f)        
        quiz.quiz_identifier = hash(str(_f) + str(_g) + str(_composite))
        # 正答の選択肢の生成
        quiz.data = [_var, _f, _g, _is_fog, _composite]
        ans = { 'fraction': 100, 'data': _composite }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_var, _f, _g, _is_fog, _composite] = quiz.data
        if _is_fog is False:
            _incorrect = _f.subs(_var, _g)
        else:
            _incorrect = _g.subs(_var, _f)
        if sympy.simplify(_incorrect - _composite) != 0:
            ans['feedback'] = r'合成関数では適用順序が重要です。それが逆になっています。'
            ans['data'] = _incorrect
            answers.append(dict(ans))          
        _incorrect = _f + _g
        if sympy.simplify(_incorrect - _composite) != 0:
            ans['feedback'] = r'合成関数は，順番に関数を適用した結果の対応を表す関数です。関数の加減算ではありません。'
            ans['data'] = _incorrect
            answers.append(dict(ans))        
        _incorrect = _f - _g     
        if sympy.simplify(_incorrect - _composite) != 0:
            ans['feedback'] = r'合成関数は，順番に関数を適用した結果の対応を表す関数です。関数の加減算ではありません。'
            ans['data'] = _incorrect
            answers.append(dict(ans))              
        _incorrect = _f * _g     
        if sympy.simplify(_incorrect - _composite) != 0:
            ans['feedback'] = r'合成関数は，順番に関数を適用した結果の対応を表す関数です。関数の乗除算ではありません。'
            ans['data'] = _incorrect
            answers.append(dict(ans))                    
        _incorrect = _f / _g     
        if sympy.simplify(_incorrect - _composite) != 0:
            ans['feedback'] = r'合成関数は，順番に関数を適用した結果の対応を表す関数です。関数の乗除算ではありません。'
            ans['data'] = _incorrect
            answers.append(dict(ans))     
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_var, _f, _g, _is_fog, _composite] = quiz.data
        _text = r'次の関数\( f(x), g(x) \)に対し，合成関数'
        if _is_fog is True:
            _text += r'\( f\circ g \)'
        else:
            _text += r'\( g\circ f \)'
        _text += r'を選択してください。<br />'
        _text += r'\( f(x)=' + sympy.latex(_f, order='lex') + r',\;g(x)=' + sympy.latex(_g, order='lex') + r' \)'
        return _text
    def answer_text(self, ans):
        return r'\( ' +  sympy.latex(ans['data'], order='lex') + r' \)'


# In[8]:


if __name__ == "__main__":
    q = composite_function()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[63]:


if __name__ == "__main__":
    pass
    #qz.save('composite_function.xml')


# ## limit of polynomial or rational function

# In[64]:


class limit_of_polynomial_or_rational_function(core.Question):
    name = '多項式又は有理関数の極限（両側極限）'
    _function_types = ['polynomial', 'rational']
    def __init__(self, emin=-3, emax=3, amin=-1, amax=1):
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 極限を求めるxの値の範囲
        self.a_min = amin
        self.a_max = amax
    def _function_generate(self, _var):
        _type = random.choice(self._function_types)
        if _type == 'polynomial':
            return sum([random.randint(self.elem_min, self.elem_max)*_var**i for i in range(0,random.randint(2,4))])
        else:
            _den = 0
            while _den == 0:
                _den = sum([random.randint(self.elem_min, self.elem_max)*_var**i for i in range(0,random.randint(2,3))])
            return sum([random.randint(self.elem_min, self.elem_max)*_var**i for i in range(0,random.randint(1,3))]) / _den
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='多項式又は有理関数の極限', quiz_number=_quiz_number)
        _var = sympy.Symbol('x')
        while True:
            _f = self._function_generate(_var)
            _a = random.choice(list(range(self.a_min, self.a_max)) + [sympy.S.Infinity, -sympy.S.Infinity])
            _limit = sympy.limit(_f, _var, _a, dir="+")
            _limit_left = sympy.limit(_f, _var, _a, dir="-")
            if _limit == _limit_left:
                break        
        quiz.quiz_identifier = hash(str(_f) + str(_a) + str(_limit))
        # 正答の選択肢の生成
        quiz.data = [_var, _f, _a, _limit]
        ans = { 'fraction': 100, 'data': _limit }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_var, _f, _a, _limit] = quiz.data
        _incorrect = (-1)*_limit
        if _incorrect != _limit:
            ans['feedback'] = r'符号を間違えています。計算を丁寧に行ってください。'
            ans['data'] = _incorrect
            answers.append(dict(ans))   
        _incorrect = sympy.limit(_f, _var, (-1)*_a, dir="+")
        if _incorrect != _limit:
            ans['feedback'] = r'符号を間違えています。計算を丁寧に行ってください。'
            ans['data'] = _incorrect
            answers.append(dict(ans))   
        for _possible_a in list(range(self.a_min, self.a_max)) + [sympy.S.Infinity, -sympy.S.Infinity]:
            _incorrect = sympy.limit(_f, _var, _possible_a, dir="+")
            if _incorrect != _limit:
                ans['feedback'] = r'近づける先が間違っています。計算を丁寧に行ってください。'
                ans['data'] = _incorrect
                answers.append(dict(ans))
        answers = common.answer_union(answers)
        while len(answers) < size:
            _incorrect = random.randint(-size, size)
            if _incorrect != _limit:
                ans['feedback'] = r'計算を丁寧に行ってください。'
                ans['data'] = _incorrect
                answers.append(dict(ans))
                answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_var, _f, _a, _limit] = quiz.data
        _text = r'次の極限を選択してください。<br />'
        _text += r'\( \lim_{x\rightarrow' + sympy.latex(_a) + r'}' + sympy.latex(_f, order='lex') + r' \)'
        return _text
    def answer_text(self, ans):
        return r'\( ' +  sympy.latex(ans['data'], order='lex') + r' \)'


# In[65]:


if __name__ == "__main__":
    q = limit_of_polynomial_or_rational_function()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[66]:


if __name__ == "__main__":
    pass
    #qz.save('limit_of_polynomial_or_rational_function.xml')


# ## limit of polynomial or rational function with no limit

# In[74]:


class limit_of_polynomial_or_rational_function_with_nolimit(core.Question):
    name = '多項式又は有理関数の極限（両側極限がないのも含む）'
    _function_types = ['polynomial', 'rational']
    def __init__(self, emin=-3, emax=3, amin=-1, amax=1):
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 極限を求めるxの値の範囲
        self.a_min = amin
        self.a_max = amax
    def _function_generate(self, _var):
        _type = random.choice(self._function_types)
        if _type == 'polynomial':
            return sum([random.randint(self.elem_min, self.elem_max)*_var**i for i in range(0,random.randint(2,4))])
        else:
            _den = 0
            while _den == 0:
                _den = sum([random.randint(self.elem_min, self.elem_max)*_var**i for i in range(0,random.randint(2,3))])
            return sum([random.randint(self.elem_min, self.elem_max)*_var**i for i in range(0,random.randint(1,3))]) / _den
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='多項式又は有理関数の極限（なしも含む）', quiz_number=_quiz_number)
        _var = sympy.Symbol('x')
        _f = self._function_generate(_var)
        _a = random.choice(list(range(self.a_min, self.a_max)) + [sympy.S.Infinity, -sympy.S.Infinity])
        _limit = sympy.limit(_f, _var, _a, dir="+")
        _limit_left = sympy.limit(_f, _var, _a, dir="-")
        if _limit != _limit_left:
            _limit = r'極限は存在しない'
            _type = r''
        elif _limit is sympy.S.Infinity or _limit is -sympy.S.Infinity:
            _type = r'発散'
        else:
            _type = r'収束'
        quiz.quiz_identifier = hash(str(_f) + str(_a) + str(_limit))
        # 正答の選択肢の生成
        quiz.data = [_var, _f, _a, _limit, _type]
        ans = { 'fraction': 100, 'data': [_limit, _type] }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_var, _f, _a, _limit, _type] = quiz.data
        _incorrect = r'極限は存在しない'
        if _incorrect != _limit:
            ans['feedback'] = r'無限大へ近づけているか，または右側極限と左側極限は一致しますので，極限は存在します。'
            ans['data'] = [_incorrect, r'']
            answers.append(dict(ans))
            if _type == r'発散':
                _wrong_type = r'収束'
            else:
                _wrong_type = r'発散'
            ans['feedback'] = r'発散とは正か負の無限大へ近づくことで，収束とはそれ以外に近づくことです。確認しましょう。'
            ans['data'] = [_limit, _wrong_type]
            answers.append(dict(ans))            
        _incorrect = (-1)*_limit
        if _incorrect != _limit:
            ans['feedback'] = r'少なくとも極限値に関して符号を間違えています。計算を丁寧に行ってください。'
            ans['data'] = [_incorrect, r'収束']
            answers.append(dict(ans))   
            ans['feedback'] = r'少なくとも極限値に関して符号を間違えています。計算を丁寧に行ってください。'
            ans['data'] = [_incorrect, r'発散']
            answers.append(dict(ans))   
        _incorrect = sympy.limit(_f, _var, (-1)*_a, dir="+")
        if _incorrect != _limit:
            ans['feedback'] = r'少なくとも極限値に関して符号を間違えています。計算を丁寧に行ってください。'
            ans['data'] = [_incorrect, r'収束']
            answers.append(dict(ans))   
            ans['feedback'] = r'少なくとも極限値に関して符号を間違えています。計算を丁寧に行ってください。'
            ans['data'] = [_incorrect, r'発散']
            answers.append(dict(ans))   
        for _possible_a in list(range(self.a_min, self.a_max)) + [sympy.S.Infinity, -sympy.S.Infinity]:
            _incorrect = sympy.limit(_f, _var, _possible_a, dir="+")
            if _incorrect != _limit:
                ans['feedback'] = r'近づける先が間違っています。計算を丁寧に行ってください。'
                ans['data'] = [_incorrect, r'収束']
                answers.append(dict(ans))
                ans['feedback'] = r'近づける先が間違っています。計算を丁寧に行ってください。'
                ans['data'] = [_incorrect, r'発散']
                answers.append(dict(ans))
        answers = common.answer_union(answers)
        while len(answers) < size:
            _incorrect = random.randint(-size, size)
            if _incorrect != _limit:
                ans['feedback'] = r'少なくとも極限値に関して計算を丁寧に行ってください。'
                ans['data'] = [_incorrect, r'収束']
                answers.append(dict(ans))
                ans['feedback'] = r'少なくとも極限値に関して計算を丁寧に行ってください。'
                ans['data'] = [_incorrect, r'発散']
                answers.append(dict(ans))
                answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_var, _f, _a, _limit, _type] = quiz.data
        _text = r'次の極限を選択してください。<br />'
        _text += r'\( \lim_{x\rightarrow' + sympy.latex(_a) + r'}' + sympy.latex(_f, order='lex') + r' \)'
        return _text
    def answer_text(self, ans):
        if isinstance(ans['data'][0], str):
            return ans['data'][0]
        else:
            return ans['data'][1] + r'し極限値は，\( ' +  sympy.latex(ans['data'][0], order='lex') + r' \)'


# In[75]:


if __name__ == "__main__":
    q = limit_of_polynomial_or_rational_function_with_nolimit()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[76]:


if __name__ == "__main__":
    pass
    #qz.save('limit_of_polynomial_or_rational_function_with_nolimit.xml')


# ## directional limit of rational function

# In[79]:


class directional_limit_of_rational_function(core.Question):
    name = '有理関数の片側極限（基本的に分母の根への片側極限）'
    def __init__(self, emin=-3, emax=3, amin=-3, amax=3):
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 極限を求めるxの値の範囲
        self.a_min = amin
        self.a_max = amax
    def _function_generate(self, _var, _a):
        _den = 0
        while _den == 0:
            _den = sympy.expand((_var - _a)*sum([random.randint(self.elem_min, self.elem_max)*_var**i for i in range(0,random.randint(1,2))]))
        if random.random() < 0.25:
            _num = abs(_var - _a) + nonzero_randint(self.elem_min, self.elem_max)*(_var - _a)
        elif random.random() < 0.25:
            _num = abs(sympy.expand((_var - _a)*(_var + _a))) + nonzero_randint(self.elem_min, self.elem_max)*(_var - _a)
        else:
            _num = sum([random.randint(self.elem_min, self.elem_max)*_var**i for i in range(0,random.randint(1,3))])
        return _num / _den
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='有理関数の片側極限', quiz_number=_quiz_number)
        _var = sympy.Symbol('x')
        while True:
            _a = random.choice(list(range(self.a_min, self.a_max)))
            _f = self._function_generate(_var, _a)
            _limit_right = sympy.limit(_f, _var, _a, dir="+")
            _limit_left = sympy.limit(_f, _var, _a, dir="-")
            if _limit_right != _limit_left:
                break        
        if random.random() < 0.5:
            _direction = "+"
            _limit = _limit_right
        else:
            _direction = "-"
            _limit = _limit_left
        quiz.quiz_identifier = hash(str(_f) + str(_a) + str(_limit))
        # 正答の選択肢の生成
        quiz.data = [_var, _f, _a, _direction, _limit]
        ans = { 'fraction': 100, 'data': _limit }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_var, _f, _a, _direction, _limit] = quiz.data
        if _direction == "+":
            _incorrect = sympy.limit(_f, _var, _a, dir="-")
        else:
            _incorrect = sympy.limit(_f, _var, _a, dir="+")
        if _incorrect != _limit:
            ans['feedback'] = r'右側極限と左側極限を取り違えている可能性があります。'
            ans['data'] = _incorrect
            answers.append(dict(ans))   
        _incorrect = (-1)*_limit
        if _incorrect != _limit:
            ans['feedback'] = r'符号を間違えています。計算を丁寧に行ってください。'
            ans['data'] = _incorrect
            answers.append(dict(ans))   
        _incorrect = sympy.limit(_f, _var, (-1)*_a, dir=_direction)
        if _incorrect != _limit:
            ans['feedback'] = r'符号を間違えています。計算を丁寧に行ってください。'
            ans['data'] = _incorrect
            answers.append(dict(ans))   
        for _possible_a in list(range(self.a_min, self.a_max)) + [sympy.S.Infinity, -sympy.S.Infinity]:
            _incorrect = sympy.limit(_f, _var, _possible_a, dir=_direction)
            if _incorrect != _limit:
                ans['feedback'] = r'近づける先が間違っています。計算を丁寧に行ってください。'
                ans['data'] = _incorrect
                answers.append(dict(ans))
        answers = common.answer_union(answers)
        while len(answers) < size:
            _incorrect = random.randint(-size, size)
            if _incorrect != _limit:
                ans['feedback'] = r'計算を丁寧に行ってください。'
                ans['data'] = _incorrect
                answers.append(dict(ans))
                answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_var, _f, _a, _direction, _limit] = quiz.data
        _text = r'次の極限を選択してください。<br />'
        _text += r'\( \lim_{x\rightarrow' + sympy.latex(_a)
        if _direction == "+":
            _text += r'+0'
        else:
            _text += r'-0'
        _text += r'}' + sympy.latex(_f, order='lex') + r' \)'
        return _text
    def answer_text(self, ans):
        return r'\( ' +  sympy.latex(ans['data'], order='lex') + r' \)'


# In[80]:


if __name__ == "__main__":
    q = directional_limit_of_rational_function()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[81]:


if __name__ == "__main__":
    pass
    #qz.save('directional_limit_of_rational_function.xml')


# ## asymptotic behavior of limit

# In[120]:


class asymptotic_behavior_of_limit(core.Question):
    name = '無限小・無限大（収束・発散の速さ）'
    _problem_types = ['lower_infty', 'higher_infty', 'higher_zero', 'same_infty', 'same_zero']
    def __init__(self, emin=-3, emax=3, dmax=3):
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 位数の最大
        self.deg_max = dmax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='無限小・無限大', quiz_number=_quiz_number)
        _var = sympy.Symbol('x')
        _deg = sympy.Rational(random.randint(1,self.deg_max*2),2)
        _type = random.choice(self._problem_types)
        quiz.quiz_identifier = hash(str(_deg) + str(_type) + str(random.random()))
        # 正答の選択肢の生成
        quiz.data = [_var, _deg, _type]
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        [_var, _deg, _type] = quiz.data
        if _type == 'lower_infty':
            _a = sympy.S.Infinity
            _lim_f = sympy.S.Infinity
            _lim_x = sympy.S.Infinity
            _lim_fx = 0
            _lim_xf = sympy.S.Infinity
        elif _type == 'higher_infty':
            _a = sympy.S.Infinity
            _lim_f = sympy.S.Infinity
            _lim_x = sympy.S.Infinity
            _lim_fx = sympy.S.Infinity
            _lim_xf = 0
        elif _type == 'same_infty':
            _a = sympy.S.Infinity
            _lim_f = sympy.S.Infinity
            _lim_x = sympy.S.Infinity
            _lim_fx = abs(nonzero_randint(self.elem_min, self.elem_max))
            _lim_xf = sympy.Rational(1, _lim_fx)
        elif _type == 'higher_zero':
            _a = 0
            _lim_f = 0
            _lim_x = 0
            _lim_fx = 0
            _lim_xf = sympy.S.Infinity
        else: #'same_zero'
            _a = 0
            _lim_f = 0
            _lim_x = 0
            _lim_fx = abs(nonzero_randint(self.elem_min, self.elem_max))
            _lim_xf = sympy.Rational(1, _lim_fx)        
        ans = { 'fraction': 100, 'data': [_a, _var, _deg, _lim_f, _lim_x, _lim_fx, _lim_xf] }
        return [ans]
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_var, _deg, _type] = quiz.data
        if _type != 'lower_infty':
            _a = sympy.S.Infinity
            _lim_f = sympy.S.Infinity
            _lim_x = sympy.S.Infinity
            _lim_fx = 0
            _lim_xf = sympy.S.Infinity
            ans['feedback'] = r'この条件では，低位の無限大が適切になります。'
            ans['data'] = [_a, _var, _deg, _lim_f, _lim_x, _lim_fx, _lim_xf]
            answers.append(dict(ans))
        if _type != 'higher_infty':
            _a = sympy.S.Infinity
            _lim_f = sympy.S.Infinity
            _lim_x = sympy.S.Infinity
            _lim_fx = sympy.S.Infinity
            _lim_xf = 0
            ans['feedback'] = r'この条件では，高位の無限大が適切になります。'
            ans['data'] = [_a, _var, _deg, _lim_f, _lim_x, _lim_fx, _lim_xf]
            answers.append(dict(ans))
        if _type != 'same_infty':
            _a = sympy.S.Infinity
            _lim_f = sympy.S.Infinity
            _lim_x = sympy.S.Infinity
            _lim_fx = abs(nonzero_randint(self.elem_min, self.elem_max))
            _lim_xf = sympy.Rational(1, _lim_fx)
            ans['feedback'] = r'この条件では，同位の無限大が適切になります。'
            ans['data'] = [_a, _var, _deg, _lim_f, _lim_x, _lim_fx, _lim_xf]
            answers.append(dict(ans))
        if _type != 'higher_zero':
            _a = 0
            _lim_f = 0
            _lim_x = 0
            _lim_fx = 0
            _lim_xf = sympy.S.Infinity
            ans['feedback'] = r'この条件では，高位の無限小が適切になります。'
            ans['data'] = [_a, _var, _deg, _lim_f, _lim_x, _lim_fx, _lim_xf]
            answers.append(dict(ans))
        if _type != 'same_zero':
            _a = 0
            _lim_f = 0
            _lim_x = 0
            _lim_fx = abs(nonzero_randint(self.elem_min, self.elem_max))
            _lim_xf = sympy.Rational(1, _lim_fx)  
            ans['feedback'] = r'この条件では，同位の無限小が適切になります。'
            ans['data'] = [_a, _var, _deg, _lim_f, _lim_x, _lim_fx, _lim_xf]
            answers.append(dict(ans))
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers
    def question_text(self, quiz):
        [_var, _deg, _type] = quiz.data
        _text = r'無限小や無限大という観点から次に最も該当する条件を選択してください。<br />『'
        if _type == 'lower_infty':
            if random.random() < 0.5:
                _text += r'\( f(x) \)は\( ' + sympy.latex(_var**_deg) + r' \)より低位の無限大'
            else:
                _text += r'\( f(x)=o(' + sympy.latex(_var**_deg) + r')\;(x\rightarrow\infty) \)'
        elif _type == 'higher_infty':
            _text += r'\( f(x) \)は\( ' + sympy.latex(_var**_deg) + r' \)より高位の無限大'
        elif _type == 'same_infty':
            if random.random() < 0.34:
                _text += r'\( f(x) \)は\( x \)に対して\( ' + sympy.latex(_deg) + r' \)位の無限大'
            elif random.random() < 0.5:
                _text += r'\( f(x)=O(' + sympy.latex(_var**_deg) + r')\;(x\rightarrow\infty) \) かつ '
                _text += r'\( ' + sympy.latex(_var**_deg) + r'=O(f(x))\;(x\rightarrow\infty) \)'
            else:
                _text += r'\( f(x) \)と\( ' + sympy.latex(_var**_deg) + r' \)は同位の無限大'                
        elif _type == 'higher_zero':
            if random.random() < 0.5:
                _text += r'\( f(x) \)は\( ' + sympy.latex(_var**_deg) + r' \)より高位の無限小'
            else:
                _text += r'\( f(x)=o(' + sympy.latex(_var**_deg) + r')\;(x\rightarrow 0) \)'            
        else: #'same_zero'
            if random.random() < 0.34:
                _text += r'\( f(x) \)は\( x \)に対して\( ' + sympy.latex(_deg) + r' \)位の無限小'
            elif random.random() < 0.5:
                _text += r'\( f(x)=O(' + sympy.latex(_var**_deg) + r')\;(x\rightarrow 0) \) かつ '
                _text += r'\( ' + sympy.latex(_var**_deg) + r'=O(f(x))\;(x\rightarrow 0) \)'
            else:
                _text += r'\( f(x) \)と\( ' + sympy.latex(_var**_deg) + r' \)は同位の無限小'
        _text += r'』'
        return _text
    def answer_text(self, ans):
        [_a, _var, _deg, _lim_f, _lim_x, _lim_fx, _lim_xf] = ans['data']
        _text = r'\( \lim_{x\rightarrow ' + sympy.latex(_a) + r'}f(x)=' + sympy.latex(_lim_f) + r',\;'
        _text += r'\lim_{x\rightarrow ' + sympy.latex(_a) + r'}' + sympy.latex(_var**_deg) + r'=' + sympy.latex(_lim_x) + r',\;'
        _text += r'\lim_{x\rightarrow ' + sympy.latex(_a) + r'}\frac{f(x)}{' + sympy.latex(_var**_deg) + r'}=' + sympy.latex(_lim_fx) + r',\;'
        _text += r'\lim_{x\rightarrow ' + sympy.latex(_a) + r'}\frac{' + sympy.latex(_var**_deg) + r'}{f(x)}=' + sympy.latex(_lim_xf) + r' \)'
        return _text


# In[121]:


if __name__ == "__main__":
    q = asymptotic_behavior_of_limit()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[122]:


if __name__ == "__main__":
    pass
    #qz.save('asymptotic_behavior_of_limit.xml')


# ## limit of trigonometric function

# In[13]:


class limit_of_trigonometric_function(core.Question):
    name = '三角関数の極限（両側極限がないのも含む）'
    _a = sympy.Symbol('a') # non-zero
    _b = sympy.Symbol('b') # non-zero
    _c = sympy.Symbol('c') # non-zero
    _s = sympy.Symbol('s') # positive
    _t = sympy.Symbol('t') # positive
    _q = sympy.Symbol('q') # poly with zero-constant
    _p = sympy.Symbol('p') # poly with zero-constant
    _x = sympy.Symbol('x')
    _major_angles = [sympy.pi/i for i in [-6,-4,-3,-2,2,3,4,6]] + [0]
    _major_angles_for_tan = [sympy.pi/i for i in [-6,-4,-3,3,4,6]] + [0]
    _skeletons = [[_a*sympy.sin(_c*_x), [sympy.S.Infinity, -sympy.S.Infinity], False], 
                  [_a*sympy.cos(_c*_x), [sympy.S.Infinity, -sympy.S.Infinity], False], 
                  [_a*sympy.sin(_c/_x), [0], False], 
                  [_a*sympy.cos(_c/_x), [0], False], 
                  [_a*sympy.sin(_x), _major_angles, True], 
                  [_a*sympy.cos(_x), _major_angles, True], 
                  [_a*sympy.tan(_x), _major_angles_for_tan, True], 
                  [_a*sympy.sin(_c/_x), [sympy.S.Infinity, -sympy.S.Infinity], True], 
                  [_a*sympy.cos(_c/_x), [sympy.S.Infinity, -sympy.S.Infinity], True], 
                  [_a*sympy.tan(_c/_x), [sympy.S.Infinity, -sympy.S.Infinity], True], 
                  [_a*_x*sympy.sin(_c/_x), [0], True], 
                  [_a*_x*sympy.cos(_c/_x), [0], True], 
                  [(_a/_x)*sympy.sin(_c*_x), [sympy.S.Infinity, -sympy.S.Infinity], True], 
                  [(_a/_x)*sympy.cos(_c*_x), [sympy.S.Infinity, -sympy.S.Infinity], True], 
                  [sympy.sin(_b*_x)/(_a*_x), [0], True], 
                  [sympy.sin(_b*_x)/(sympy.sin(_a*_x)), [0], True], 
                  [sympy.tan(_b*_x)/(_a*_x), [0], True], 
                  [sympy.sin(_b*_x**_s)/(_a*_x**_t), [0], True], 
                  [sympy.sin(_q)/(_p), [0], True]]
    def __init__(self, emin=-5, emax=5, dmax=3):
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 多項式の次数の最大
        self.deg_max = dmax
    def _function_generate(self, _func):
        _func = _func.subs(self._a, nonzero_randint(self.elem_min, self.elem_max))
        _func = _func.subs(self._b, nonzero_randint(self.elem_min, self.elem_max))
        _func = _func.subs(self._c, nonzero_randint(self.elem_min, self.elem_max))
        _func = _func.subs(self._s, abs(nonzero_randint(self.elem_min, self.elem_max)))
        _func = _func.subs(self._t, abs(nonzero_randint(self.elem_min, self.elem_max)))
        _poly = 0
        while _poly == 0:
            _poly = sum([random.randint(self.elem_min, self.elem_max)*self._x**i for i in range(1,random.randint(1,self.deg_max))])
        _func = _func.subs(self._p, _poly)
        _poly = 0
        while _poly == 0:
            _poly = sum([random.randint(self.elem_min, self.elem_max)*self._x**i for i in range(1,random.randint(1,self.deg_max))])
        _func = _func.subs(self._q, _poly)
        return _func
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='三角関数の極限（なしも含む）', quiz_number=_quiz_number)
        [_func, _alist, _exists] = random.choice(self._skeletons)
        _func = self._function_generate(_func)
        _a = random.choice(_alist)
        _limit = r'not exist'
        if _exists:
            _limit = sympy.limit(_func, self._x, _a, dir="+")
            _limit_left = sympy.limit(_func, self._x, _a, dir="-")
            if _limit != _limit_left:
                _exists = False
        quiz.quiz_identifier = hash(str(_func) + str(_a) + str(_limit))
        # 正答の選択肢の生成
        quiz.data = [_func, _a, _exists, _limit]
        ans = { 'fraction': 100, 'data': [_exists, _limit] }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_func, _a, _exists, _limit] = quiz.data
        if _exists:
            _incorrect = [False, _limit]
            ans['feedback'] = r'無限大へ近づけているか，または右側極限と左側極限は一致しますので，極限は存在します。'
            ans['data'] = _incorrect
            answers.append(dict(ans))
        ans['data'] = [True, sympy.S.Infinity]
        if not _exists:
            ans['feedback'] = r'極限の状況には，収束・発散・存在しない，の3通りあります。発散していませんので，よく確認しましょう。'
            answers.append(dict(ans))    
        elif abs(_limit) != sympy.S.Infinity:
            ans['feedback'] = r'極限の状況には，収束・発散・存在しない，の3通りあります。発散していませんので，よく確認しましょう。'
            answers.append(dict(ans))    
        elif _limit != sympy.S.Infinity:
            ans['feedback'] = r'極限を求める場合，符号のミスは起こりやすいものの1つです。よく確認しましょう。'
            answers.append(dict(ans))    
        ans['data'] = [True, -sympy.S.Infinity]
        if not _exists:
            ans['feedback'] = r'極限の状況には，収束・発散・存在しない，の3通りあります。発散していませんので，よく確認しましょう。'
            answers.append(dict(ans))    
        elif abs(_limit) != sympy.S.Infinity:
            ans['feedback'] = r'極限の状況には，収束・発散・存在しない，の3通りあります。発散していませんので，よく確認しましょう。'
            answers.append(dict(ans))    
        elif _limit != -sympy.S.Infinity:
            ans['feedback'] = r'極限を求める場合，符号のミスは起こりやすいものの1つです。よく確認しましょう。'
            answers.append(dict(ans))    
        ans['data'] = [True, 0]
        if not _exists:
            ans['feedback'] = r'極限の状況には，収束・発散・存在しない，の3通りあります。収束していませんので，よく確認しましょう。'
            answers.append(dict(ans))    
        elif abs(_limit) == sympy.S.Infinity:
            ans['feedback'] = r'極限の状況には，収束・発散・存在しない，の3通りあります。収束していませんので，よく確認しましょう。'
            answers.append(dict(ans))    
        elif _limit != 0:
            ans['feedback'] = r'極限を求める場合，最後の数値の四則演算などで間違えることもあります。よく確認しましょう。'
            answers.append(dict(ans))
        for i in range(2):
            _incorrect_limit = nonzero_randint(self.elem_min, self.elem_max)
            ans['data'] = [True, _incorrect_limit]
            if not _exists:
                ans['feedback'] = r'極限の状況には，収束・発散・存在しない，の3通りあります。収束していませんので，よく確認しましょう。'
                answers.append(dict(ans))    
            elif abs(_limit) == sympy.S.Infinity:
                ans['feedback'] = r'極限の状況には，収束・発散・存在しない，の3通りあります。収束していませんので，よく確認しましょう。'
                answers.append(dict(ans))    
            elif _limit != _incorrect_limit:
                ans['feedback'] = r'極限を求める場合，最後の数値の四則演算などで間違えることもあります。よく確認しましょう。'
                answers.append(dict(ans))  
        for i in range(2):  
            _incorrect_limit = sympy.Rational(nonzero_randint(self.elem_min, self.elem_max), nonzero_randint(self.elem_min, self.elem_max))
            ans['data'] = [True, _incorrect_limit]
            if not _exists:
                ans['feedback'] = r'極限の状況には，収束・発散・存在しない，の3通りあります。収束していませんので，よく確認しましょう。'
                answers.append(dict(ans))    
            elif abs(_limit) == sympy.S.Infinity:
                ans['feedback'] = r'極限の状況には，収束・発散・存在しない，の3通りあります。収束していませんので，よく確認しましょう。'
                answers.append(dict(ans))    
            elif _limit != _incorrect_limit:
                ans['feedback'] = r'極限を求める場合，最後の数値の四則演算などで間違えることもあります。よく確認しましょう。'
                answers.append(dict(ans))    
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_func, _a, _exists, _limit] = quiz.data
        _text = r'次の極限を選択してください。<br />'
        _text += r'\( \lim_{' + sympy.latex(self._x) + r'\rightarrow' + sympy.latex(_a) + r'}' + sympy.latex(_func, order='lex') + r' \)'
        return _text
    def answer_text(self, ans):
        if ans['data'][0]:
            return r'\( ' +  sympy.latex(ans['data'][1], order='lex') + r' \)'
        else:
            return r'極限は存在しない'


# In[14]:


if __name__ == "__main__":
    q = limit_of_trigonometric_function()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[15]:


if __name__ == "__main__":
    pass
    #qz.save('limit_of_trigonometric_function.xml')


# ## limit of exponential and logarithmic function

# In[85]:


class limit_of_exponential_and_logarithmic_function(core.Question):
    name = '指数関数と対数関数の極限'
    _a = sympy.Symbol('a') # non-ONE positive rational (could be less than 1) or integer (larger than 1)
    _b = sympy.Symbol('b') # non-zero integer
    _c = sympy.Symbol('c') # positive integer
    _d = sympy.Symbol('d') # non-ONE positive rational (could be less than 1) or integer (larger than 1)
    _e = sympy.Symbol('e') # non-zero integer
    _f = sympy.Symbol('f') # positive integer
    _x = sympy.Symbol('x')
    # [expression, [x tends to, direction], [tex form of expresion: string or expression]]
    _skeletons = [[_a**(_b*_x), [[sympy.S.Infinity,0], [-sympy.S.Infinity,0], [0,0]], [_a**(_b*_x)]], 
                  [sympy.log(_c*_x, _a), [[sympy.S.Infinity,0], [0,+1]], [r'\log_{', _a, r'}(', _c*_x, r')']], 
                  [_a**(_b*_x)-_d**(_e*_x), [[sympy.S.Infinity,0], [-sympy.S.Infinity,0], [0,0]], [_a**(_b*_x)-_d**(_e*_x)]], 
#                  [sympy.log(_c*_x+_b, _a)-sympy.log(_f*_x+_e, _a), [[sympy.S.Infinity,0]], [r'\log_{', _a, r'}(', _c*_x+_b, r')-\log_{', _a, r'}(', _f*_x+_e, r')']], 
                  [(1+_b/_x)**(_e*_x), [[sympy.S.Infinity,0], [-sympy.S.Infinity,0]], [(1+_b/_x)**(_e*_x)]], 
                  [(_b/_x)*sympy.log(_c*_x+1), [[0,0]], [(_b/_x)*sympy.log(_c*_x+1)]], 
                  [(sympy.exp(_b*_x)-1)/(_e*_x), [[0,0]], [(sympy.exp(_b*_x)-1)/(_e*_x)]]]
    def __init__(self, emin=-3, emax=3):
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
    def _parameter_substitution(self, _func, _values):
        _func = _func.subs(self._a, _values['a'])
        _func = _func.subs(self._b, _values['b'])
        _func = _func.subs(self._c, _values['c'])
        _func = _func.subs(self._d, _values['d'])
        _func = _func.subs(self._e, _values['e'])
        _func = _func.subs(self._f, _values['f'])
        return _func
    def _generate_values(self):
        _values = dict()
        _values['a'] = sympy.Rational(1,nonone_randpint(self.elem_min,self.elem_max)) if random.random() < 0.5 else nonone_randpint(self.elem_min,self.elem_max)
        _values['b'] = nonzero_randint(self.elem_min,self.elem_max)
        _values['c'] = abs(nonzero_randint(self.elem_min,self.elem_max))
        _values['d'] = sympy.Rational(1,nonone_randpint(self.elem_min,self.elem_max)) if random.random() < 0.5 else nonone_randpint(self.elem_min,self.elem_max)
        _values['e'] = nonzero_randint(self.elem_min,self.elem_max)
        _values['f'] = abs(nonzero_randint(self.elem_min,self.elem_max))
        return _values
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='指数関数と対数関数の極限', quiz_number=_quiz_number)
        while True:
            [_funcO, _alist, _texform] = random.choice(self._skeletons)
            _values = self._generate_values()
            _func = self._parameter_substitution(_funcO, _values)
            if not _func.is_number:
                break
        [_a, _dir] = random.choice(_alist)
#        print(_funcO, " : ", _func, ":", _a, " with ", _values)
        if _dir == 0:
            _limit = sympy.limit(sympy.simplify(_func), self._x, _a, dir="+-")
        elif _dir > 0:
            _limit = sympy.limit(sympy.simplify(_func), self._x, _a, dir="+")
        else:
            _limit = sympy.limit(sympy.simplify(_func), self._x, _a, dir="-")
#        print('===> ', _limit)
        quiz.quiz_identifier = hash(str(_func) + str(_a) + str(_limit))
        # 正答の選択肢の生成
        quiz.data = [_func, _a, _dir, _limit, _values, _texform]
        ans = { 'fraction': 100, 'data': _limit }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_func, _a, _dir, _limit, _values, _texform] = quiz.data
        ans['feedback'] = r'無限大へ近づけているか，または右側極限と左側極限は一致しますので，極限は存在します。'
        ans['data'] = r'not exist'
        answers.append(dict(ans))
        ans['data'] = sympy.S.Infinity
        if abs(_limit) != sympy.S.Infinity:
            ans['feedback'] = r'極限の状況には，収束・発散・存在しない，の3通りあります。発散していませんので，よく確認しましょう。'
            answers.append(dict(ans))    
        elif _limit != sympy.S.Infinity:
            ans['feedback'] = r'極限を求める場合，符号のミスは起こりやすいものの1つです。よく確認しましょう。'
            answers.append(dict(ans))    
        ans['data'] = -sympy.S.Infinity
        if abs(_limit) != sympy.S.Infinity:
            ans['feedback'] = r'極限の状況には，収束・発散・存在しない，の3通りあります。発散していませんので，よく確認しましょう。'
            answers.append(dict(ans))    
        elif _limit != -sympy.S.Infinity:
            ans['feedback'] = r'極限を求める場合，符号のミスは起こりやすいものの1つです。よく確認しましょう。'
            answers.append(dict(ans))    
        ans['data'] = 0
        if abs(_limit) == sympy.S.Infinity:
            ans['feedback'] = r'極限の状況には，収束・発散・存在しない，の3通りあります。収束していませんので，よく確認しましょう。'
            answers.append(dict(ans))    
        elif _limit != 0:
            ans['feedback'] = r'極限を求める場合，最後の数値の四則演算などで間違えることもあります。よく確認しましょう。'
            answers.append(dict(ans))
        if _limit != 0 and abs(_limit) != sympy.S.Infinity:
            ans['data'] = 1/_limit
            ans['feedback'] = r'極限を求める場合，式の取り違いも起こりえます。よく確認しましょう。'
            answers.append(dict(ans))    
        _incorrect_limit = nonzero_randint(self.elem_min, self.elem_max)
        ans['data'] = _incorrect_limit
        if abs(_limit) == sympy.S.Infinity:
            ans['feedback'] = r'極限の状況には，収束・発散・存在しない，の3通りあります。収束していませんので，よく確認しましょう。'
            answers.append(dict(ans))    
        elif _limit != _incorrect_limit:
            ans['feedback'] = r'極限を求める場合，最後の数値の四則演算などで間違えることもあります。よく確認しましょう。'
            answers.append(dict(ans))  
        ans['data'] = sympy.Rational(1,_incorrect_limit)
        if abs(_limit) == sympy.S.Infinity:
            ans['feedback'] = r'極限の状況には，収束・発散・存在しない，の3通りあります。収束していませんので，よく確認しましょう。'
            answers.append(dict(ans))    
        elif _limit != _incorrect_limit:
            ans['feedback'] = r'極限を求める場合，最後の数値の四則演算などで間違えることもあります。よく確認しましょう。'
            answers.append(dict(ans))  
        _incorrect_limit = sympy.Rational(nonzero_randint(self.elem_min, self.elem_max), nonzero_randint(self.elem_min, self.elem_max))
        ans['data'] = _incorrect_limit
        if abs(_limit) == sympy.S.Infinity:
            ans['feedback'] = r'極限の状況には，収束・発散・存在しない，の3通りあります。収束していませんので，よく確認しましょう。'
            answers.append(dict(ans))    
        elif _limit != _incorrect_limit:
            ans['feedback'] = r'極限を求める場合，最後の数値の四則演算などで間違えることもあります。よく確認しましょう。'
            answers.append(dict(ans))    
        ans['data'] = 1/_incorrect_limit
        if abs(_limit) == sympy.S.Infinity:
            ans['feedback'] = r'極限の状況には，収束・発散・存在しない，の3通りあります。収束していませんので，よく確認しましょう。'
            answers.append(dict(ans))    
        elif _limit != _incorrect_limit:
            ans['feedback'] = r'極限を求める場合，最後の数値の四則演算などで間違えることもあります。よく確認しましょう。'
            answers.append(dict(ans))     
        _incorrect_limit = sympy.E**nonzero_randint(self.elem_min, self.elem_max)
        ans['data'] = _incorrect_limit
        if abs(_limit) == sympy.S.Infinity:
            ans['feedback'] = r'極限の状況には，収束・発散・存在しない，の3通りあります。収束していませんので，よく確認しましょう。'
            answers.append(dict(ans))    
        elif _limit != _incorrect_limit:
            ans['feedback'] = r'極限を求める場合，最後の数値の四則演算などで間違えることもあります。よく確認しましょう。'
            answers.append(dict(ans))    
        ans['data'] = 1/_incorrect_limit
        if abs(_limit) == sympy.S.Infinity:
            ans['feedback'] = r'極限の状況には，収束・発散・存在しない，の3通りあります。収束していませんので，よく確認しましょう。'
            answers.append(dict(ans))    
        elif _limit != _incorrect_limit:
            ans['feedback'] = r'極限を求める場合，最後の数値の四則演算などで間違えることもあります。よく確認しましょう。'
            answers.append(dict(ans))    
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_func, _a, _dir, _limit, _values, _texform] = quiz.data
        _text = r'次の極限を選択してください。<br />'
        _text += r'\( \lim_{' + sympy.latex(self._x) + r'\rightarrow'
        if _dir > 0:
            _text += r'+'
        elif _dir < 0:
            _text += r'-'
        _text += sympy.latex(_a) + r'}'
        for _v in _texform:
            if isinstance(_v, str):
                _text += _v
            else:
                _text += sympy.latex(self._parameter_substitution(_v, _values), order='lex') 
        _text += r' \)'
        return _text
    def answer_text(self, ans):
        if not isinstance(ans['data'],str):
            return r'\( ' +  sympy.latex(ans['data'], order='lex') + r' \)'
        else:
            return r'極限は存在しない'


# In[86]:


if __name__ == "__main__":
    q = limit_of_exponential_and_logarithmic_function()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[87]:


if __name__ == "__main__":
    pass
    #qz.save('limit_of_exponential_and_logarithmic_function.xml')


# ## average rate of change

# In[48]:


class average_rate_of_change(core.Question):
    name = '平均変化率（微分係数の導入前段階）'
    _a = sympy.Symbol('a') # non-zero
    _b = sympy.Symbol('b') # non-zero
    _c = sympy.Symbol('c') # positive
    _x = sympy.Symbol('x')
    _skeletons = [_a*sympy.sin(_b*_x),_a*sympy.cos(_b*_x),_a*sympy.tan(_b*_x),
                  _a*_x**_c, _a/_x**_c, _a*_x+_b, 1/(_x+_a), _a*_x**2+_b, 1/(_x**2+_a)]
    def __init__(self, emin=-3, emax=3, hmin=-2, hmax=2):
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        self.h_min = hmin
        self.h_max = hmax
    def _function_generate(self, _func):
        _func = _func.subs(self._a, nonzero_randint(self.elem_min, self.elem_max))
        _func = _func.subs(self._b, nonzero_randint(self.elem_min, self.elem_max))
        _func = _func.subs(self._c, abs(nonzero_randint(self.elem_min, self.elem_max)))
        return _func
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='平均変化率', quiz_number=_quiz_number)
        _func = random.choice(self._skeletons)
        _func = self._function_generate(_func)
        _a = random.randint(self.elem_min, self.elem_max)
        _h = nonzero_randint(self.h_min, self.h_max)
        while True:
            err = 0
            if _h < 0:
                for _b in range(0, _h - 1, -1):
                    if not _func.subs(self._x, _a + _b).is_real:
                        err = err + 1
            else:
                for _b in range(0, _h + 1, 1):
                    if not _func.subs(self._x, _a + _b).is_real:
                        err = err + 1
            if err == 0:
                break
            _func = random.choice(self._skeletons)
            _func = self._function_generate(_func)
            _a = random.randint(self.elem_min, self.elem_max)
            _h = nonzero_randint(self.h_min, self.h_max)
        quiz.quiz_identifier = hash(str(_func) + str(_a) + str(_h))
        # 正答の選択肢の生成
        _arc = (_func.subs(self._x, _a + _h) - _func.subs(self._x, _a))*sympy.Rational(1,_h)
        quiz.data = [_func, _a, _h, _arc]
        ans = { 'fraction': 100, 'data': _arc }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_func, _a, _h, _arc] = quiz.data
        _incorrect = -(_func.subs(self._x, _a + _h) - _func.subs(self._x, _a))*sympy.Rational(1,_h)
        if _arc != _incorrect:
            ans['feedback'] = r'平均変化率は，変化先から変化元を引いた関数の値の変化の変数の値の変化に対する割合です。'
            ans['data'] = _incorrect
            answers.append(dict(ans))
        _incorrect = (_func.subs(self._x, _a + _h) + _func.subs(self._x, _a))*sympy.Rational(1,_h)
        if _arc != _incorrect:
            ans['feedback'] = r'平均変化率は，変化先から変化元を引いた関数の値の変化の変数の値の変化に対する割合です。'
            ans['data'] = _incorrect
            answers.append(dict(ans))
        _incorrect = -(_func.subs(self._x, _a + _h) + _func.subs(self._x, _a))*sympy.Rational(1,_h)
        if _arc != _incorrect:
            ans['feedback'] = r'平均変化率は，変化先から変化元を引いた関数の値の変化の変数の値の変化に対する割合です。'
            ans['data'] = _incorrect
            answers.append(dict(ans))
        _incorrect = (_func.subs(self._x, _a + _h) - _func.subs(self._x, _a))
        if _arc != _incorrect:
            ans['feedback'] = r'平均変化率は，変化先から変化元を引いた関数の値の変化の変数の値の変化に対する割合です。'
            ans['data'] = _incorrect
            answers.append(dict(ans))
        _incorrect = -(_func.subs(self._x, _a + _h) - _func.subs(self._x, _a))
        if _arc != _incorrect:
            ans['feedback'] = r'平均変化率は，変化先から変化元を引いた関数の値の変化の変数の値の変化に対する割合です。'
            ans['data'] = _incorrect
            answers.append(dict(ans))          
        answers = common.answer_union(answers)
        while len(answers) < size:
            _incorrect = _arc + random.randint(self.elem_min, self.elem_max)
            if _arc != _incorrect:
                ans['feedback'] = r'平均変化率は，変化先から変化元を引いた関数の値の変化の変数の値の変化に対する割合です。'
                ans['data'] = _incorrect
                answers.append(dict(ans))
            if _arc != -_incorrect:
                ans['feedback'] = r'平均変化率は，変化先から変化元を引いた関数の値の変化の変数の値の変化に対する割合です。'
                ans['data'] = -_incorrect
                answers.append(dict(ans))                
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_func, _a, _h, _arc] = quiz.data
        _text = r'次の関数\( f(x) \)が\( x=' + sympy.latex(_a) + r' \)から\( x='
        _text += sympy.latex(_a)
        if _h < 0:
            _text += r'+(' + sympy.latex(_h) + r')'
        else:
            _text += r'+' + sympy.latex(_h)
        _text += r' \)まで変化するときの平均変化率を選択してください。<br />'
        _text += r'\( f(x)=' + sympy.latex(_func, order='lex') + r' \)'
        return _text
    def answer_text(self, ans):
        return r'\( ' +  sympy.latex(ans['data'], order='lex') + r' \)'


# In[49]:


if __name__ == "__main__":
    q = average_rate_of_change()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[50]:


if __name__ == "__main__":
    pass
    #qz.save('average_rate_of_change.xml')


# ## differential coefficient

# In[5]:


class differential_coefficient(core.Question):
    name = '微分係数の計算（多項式，有理関数及び区分的関数）'
    _function_types = ['polynomial', 'rational', 'piecewise']
    _h = sympy.Symbol('h')
    def __init__(self, emin=-3, emax=3, amin=-2, amax=2):
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 極限を求めるxの値の範囲
        self.a_min = amin
        self.a_max = amax
    def _differential_coefficient(self, _func, _var, _a, _dir):
        _rate_of_change = (_func.subs(_var, _a + self._h) - _func.subs(_var, _a))/self._h
        return sympy.limit(_rate_of_change, self._h, 0, dir=_dir)
    def _function_generate(self, _var, _a):
        _type = random.choice(self._function_types)
        if _type == 'polynomial':
            _pol = sum([random.randint(self.elem_min, self.elem_max)*_var**i for i in range(0,random.randint(2,3))])
            while _pol == 0:
                _pol = sum([random.randint(self.elem_min, self.elem_max)*_var**i for i in range(0,random.randint(2,3))])
            return _pol
        elif _type == 'rational':
            _den = 0
            while _den == 0:
                _den = sum([random.randint(self.elem_min, self.elem_max)*_var**i for i in range(0,random.randint(2,3))])
            return sum([random.randint(self.elem_min, self.elem_max)*_var**i for i in range(0,random.randint(1,3))]) / _den
        else:
            _ok = random.choice([False, True])
            _notequal = _ok
            _upper = sum([random.randint(self.elem_min, self.elem_max)*_var**i for i in range(0,random.randint(2,3))])
            _lower = sum([random.randint(self.elem_min, self.elem_max)*_var**i for i in range(0,random.randint(2,3))])
            while _upper.subs(_var, _a) != _lower.subs(_var, _a):
                _upper = sum([random.randint(self.elem_min, self.elem_max)*_var**i for i in range(0,random.randint(2,3))])
                _lower = sum([random.randint(self.elem_min, self.elem_max)*_var**i for i in range(0,random.randint(2,3))])
            if _ok:
                _limit = self._differential_coefficient(_upper, _var, _a, "+")
                _limit_left = self._differential_coefficient(_lower, _var, _a, "-")
                if _limit == _limit_left:
                    _notequal = False                
            while _upper == _lower or _upper == 0 or _lower == 0 or _notequal:
                _notequal = _ok
                _upper = sum([random.randint(self.elem_min, self.elem_max)*_var**i for i in range(0,random.randint(2,3))])
                _lower = sum([random.randint(self.elem_min, self.elem_max)*_var**i for i in range(0,random.randint(2,3))])
                while _upper.subs(_var, _a) != _lower.subs(_var, _a):
                    _upper = sum([random.randint(self.elem_min, self.elem_max)*_var**i for i in range(0,random.randint(2,3))])
                    _lower = sum([random.randint(self.elem_min, self.elem_max)*_var**i for i in range(0,random.randint(2,3))])
                if _ok:
                    _limit = self._differential_coefficient(_upper, _var, _a, "+")
                    _limit_left = self._differential_coefficient(_lower, _var, _a, "-")
                    if _limit == _limit_left:
                        _notequal = False                
            return [_upper, _lower]
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='微分係数の計算（区分的関数含む）', quiz_number=_quiz_number)
        _var = sympy.Symbol('x')
        _a = random.randint(self.a_min, self.a_max)
        _f = self._function_generate(_var, _a)
        if isinstance(_f, list):
            _limit_right = self._differential_coefficient(_f[0], _var, _a, "+")
            _limit_left = self._differential_coefficient(_f[1], _var, _a, "-")
        else:
            _limit_right = self._differential_coefficient(_f, _var, _a, "+")
            _limit_left = self._differential_coefficient(_f, _var, _a, "-")
        if _limit_right != _limit_left:
            _limit = r'微分可能でない'
        else:
            _limit = _limit_right
        while True:
            if isinstance(_limit, str):
                break
            elif _limit.is_real:
                break
            _a = random.randint(self.a_min, self.a_max)
            _f = self._function_generate(_var, _a)
            if isinstance(_f, list):
                _limit_right = self._differential_coefficient(_f[0], _var, _a, "+")
                _limit_left = self._differential_coefficient(_f[1], _var, _a, "-")
            else:
                _limit_right = self._differential_coefficient(_f, _var, _a, "+")
                _limit_left = self._differential_coefficient(_f, _var, _a, "-")
            if _limit_right != _limit_left:
                _limit = r'微分可能でない'
            else:
                _limit = _limit_right            
        quiz.quiz_identifier = hash(str(_f) + str(_a) + str(_limit))
        # 正答の選択肢の生成
        quiz.data = [_var, _f, _a, _limit, _limit_right, _limit_left]
        ans = { 'fraction': 100, 'data': _limit }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_var, _f, _a, _limit, _limit_right, _limit_left] = quiz.data
        _incorrect = r'微分可能でない'
        if _incorrect != _limit:
            ans['feedback'] = r'微分係数の定義に基づき右側極限と左側極限を求めると一致しますので，微分係数は存在します。'
            ans['data'] = _incorrect
            answers.append(dict(ans))
            _incorrect = -_limit
            if _incorrect != _limit:
                ans['feedback'] = r'微分係数の定義に基づき丁寧に計算し直してください。'
                ans['data'] = _incorrect
                answers.append(dict(ans))
            _incorrect = 0
            if _incorrect != _limit:
                ans['feedback'] = r'微分係数の定義に基づき，きちんと計算をしてください。'
                ans['data'] = _incorrect
                answers.append(dict(ans))
            _incorrect = _a
            if _incorrect != _limit:
                ans['feedback'] = r'微分係数の定義に基づき，きちんと計算をしてください。'
                ans['data'] = _incorrect
                answers.append(dict(ans))
            _incorrect = -_a
            if _incorrect != _limit:
                ans['feedback'] = r'微分係数の定義に基づき，きちんと計算をしてください。'
                ans['data'] = _incorrect
                answers.append(dict(ans))
        else:
            ans['feedback'] = r'微分係数の定義に基づき極限を求めたとき，右側極限と左側極限が一致しなければ，微分係数は存在しません。'
            ans['data'] = _limit_right
            answers.append(dict(ans))
            ans['data'] = _limit_left
            answers.append(dict(ans))
            ans['data'] = -_limit_right
            answers.append(dict(ans))
            ans['data'] = -_limit_left
            answers.append(dict(ans))
            ans['feedback'] = r'微分係数の定義に基づき，きちんと計算をしてください。'
            ans['data'] = _a
            answers.append(dict(ans))
            ans['data'] = -_a
            answers.append(dict(ans))
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_var, _f, _a, _limit, _limit_right, _limit_left] = quiz.data
        _text = r'次の関数\( f(' + sympy.latex(_var) + r') \)の\( ' + sympy.latex(_var) + r'=' + sympy.latex(_a) + r' \)での微分係数を選択してください。<br />'
        _text += r'\( f(' + sympy.latex(_var) +  r')=' 
        if isinstance(_f, list):
            _text += r'\left\{\begin{array}{cc}' 
            _text += sympy.latex(_f[0], order='lex') + r'&(' + sympy.latex(_var) + r'\geq ' + sympy.latex(_a) + r')\\'
            _text += sympy.latex(_f[1], order='lex') + r'&(' + sympy.latex(_var) + r'<' + sympy.latex(_a) + r')'
            _text += r'\end{array}\right.'
        else:
            _text += sympy.latex(_f, order='lex')
        _text += r' \)'
        return _text
    def answer_text(self, ans):
        if isinstance(ans['data'], str):
            return ans['data']
        else:
            return r'\( ' + sympy.latex(ans['data']) + r' \)'


# In[6]:


if __name__ == "__main__":
    q = differential_coefficient()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=25)


# In[7]:


if __name__ == "__main__":
    #pass
    qz.save('differential_coefficient.xml')


# ## differential coefficient intermediate expression

# In[100]:


class differential_coefficient_intermediate_expression(core.Question):
    name = '微分係数の途中式（不適切なものを選択）'
    _function_types = ['polynomial', 'rational', 'piecewise']
    _h = sympy.Symbol('h')
    def __init__(self, emin=-3, emax=3, amin=-2, amax=2):
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 極限を求めるxの値の範囲
        self.a_min = amin
        self.a_max = amax
    def _differential_coefficient(self, _func, _var, _a, _dir):
        _rate_of_change = (_func.subs(_var, _a + self._h) - _func.subs(_var, _a))/self._h
        return sympy.limit(_rate_of_change, self._h, 0, dir=_dir)
    def _function_generate(self, _var, _a):
        _type = random.choice(self._function_types)
        if _type == 'polynomial':
            _pol = sum([random.randint(self.elem_min, self.elem_max)*_var**i for i in range(0,random.randint(2,3))])
            while _pol == 0:
                _pol = sum([random.randint(self.elem_min, self.elem_max)*_var**i for i in range(0,random.randint(2,3))])
            return _pol
        elif _type == 'rational':
            _den = 0
            while _den == 0:
                _den = sum([random.randint(self.elem_min, self.elem_max)*_var**i for i in range(0,random.randint(2,3))])
            return sum([random.randint(self.elem_min, self.elem_max)*_var**i for i in range(0,random.randint(1,3))]) / _den
        else:
            _ok = random.choice([False, True])
            _notequal = _ok
            _upper = sum([random.randint(self.elem_min, self.elem_max)*_var**i for i in range(0,random.randint(2,3))])
            _lower = sum([random.randint(self.elem_min, self.elem_max)*_var**i for i in range(0,random.randint(2,3))])
            while _upper.subs(_var, _a) != _lower.subs(_var, _a):
                _upper = sum([random.randint(self.elem_min, self.elem_max)*_var**i for i in range(0,random.randint(2,3))])
                _lower = sum([random.randint(self.elem_min, self.elem_max)*_var**i for i in range(0,random.randint(2,3))])
            if _ok:
                _limit = self._differential_coefficient(_upper, _var, _a, "+")
                _limit_left = self._differential_coefficient(_lower, _var, _a, "-")
                if _limit == _limit_left:
                    _notequal = False                
            while _upper == _lower or _upper == 0 or _lower == 0 or _notequal:
                _notequal = _ok
                _upper = sum([random.randint(self.elem_min, self.elem_max)*_var**i for i in range(0,random.randint(2,3))])
                _lower = sum([random.randint(self.elem_min, self.elem_max)*_var**i for i in range(0,random.randint(2,3))])
                while _upper.subs(_var, _a) != _lower.subs(_var, _a):
                    _upper = sum([random.randint(self.elem_min, self.elem_max)*_var**i for i in range(0,random.randint(2,3))])
                    _lower = sum([random.randint(self.elem_min, self.elem_max)*_var**i for i in range(0,random.randint(2,3))])
                if _ok:
                    _limit = self._differential_coefficient(_upper, _var, _a, "+")
                    _limit_left = self._differential_coefficient(_lower, _var, _a, "-")
                    if _limit == _limit_left:
                        _notequal = False                
            return [_upper, _lower]
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='微分係数の途中式', quiz_number=_quiz_number)
        _var = sympy.Symbol('x')
        _a = random.randint(self.a_min, self.a_max)
        _f = self._function_generate(_var, _a)
        quiz.quiz_identifier = hash(str(_f) + str(_a))
        quiz.data = [_var, _f, _a]
        # 不正解の選択肢の生成
        ans = dict()
        ans['fraction'] = 0
        ans['feedback'] = r'微分係数の定義に基づいた式であり，必要な極限計算となります。'
        ans['data'] = r'\( \lim_{' + sympy.latex(_var) + r'\rightarrow' + sympy.latex(_a) + r'}\frac{f(' + sympy.latex(_var) + r')-f(' + sympy.latex(_a) + r')}{' + sympy.latex(_var)
        if _a < 0:
            ans['data'] += r'+' + sympy.latex(-_a) + r'} \)'
        elif _a == 0:
            ans['data'] += r'} \)'
        else:
            ans['data'] += r'-' + sympy.latex(_a) + r'} \)'
        quiz.answers.append(dict(ans))
        if _a != 0:
            ans['data'] = r'\( \lim_{h\rightarrow 0}\frac{f(' + sympy.latex(_a) + r'+h)-f(' + sympy.latex(_a) + r')}{h} \)'
        else:
            ans['data'] = r'\( \lim_{h\rightarrow 0}\frac{f(h)-f(' + sympy.latex(_a) + r')}{h} \)'
        quiz.answers.append(dict(ans))
        ans['feedback'] = r'極限計算は丁寧に行うならば，片側極限同士の一致を調べるため必要な計算となります。'
        if _a != 0:
            ans['data'] = r'\( \lim_{' + sympy.latex(_var) + r'\rightarrow' + sympy.latex(_a) + r'-0}\frac{f(' + sympy.latex(_var) + r')-f(' + sympy.latex(_a) + r')}{' + sympy.latex(_var)
            if _a < 0:
                ans['data'] += r'+' + sympy.latex(-_a) + r'} \)'
            elif _a == 0:
                ans['data'] += r'} \)'
            else:
                ans['data'] += r'-' + sympy.latex(_a) + r'} \)'
            quiz.answers.append(dict(ans))
            ans['data'] = r'\( \lim_{' + sympy.latex(_var) + r'\rightarrow' + sympy.latex(_a) + r'+0}\frac{f(' + sympy.latex(_var) + r')-f(' + sympy.latex(_a) + r')}{' + sympy.latex(_var)
            if _a < 0:
                ans['data'] += r'+' + sympy.latex(-_a) + r'} \)'
            elif _a == 0:
                ans['data'] += r'} \)'
            else:
                ans['data'] += r'-' + sympy.latex(_a) + r'} \)'
            quiz.answers.append(dict(ans))
        else:
            ans['data'] = r'\( \lim_{' + sympy.latex(_var) + r'\rightarrow -0}\frac{f(' + sympy.latex(_var) + r')-f(' + sympy.latex(_a) + r')}{' + sympy.latex(_var) + r'} \)'
            quiz.answers.append(dict(ans))
            ans['data'] = r'\( \lim_{' + sympy.latex(_var) + r'\rightarrow +0}\frac{f(' + sympy.latex(_var) + r')-f(' + sympy.latex(_a) + r')}{' + sympy.latex(_var) + r'} \)'
            quiz.answers.append(dict(ans))
        ans['data'] = r'\( \lim_{h\rightarrow -0}\frac{f(' + sympy.latex(_a) + r'+h)-f(' + sympy.latex(_a) + r')}{h} \)'
        quiz.answers.append(dict(ans))
        ans['data'] = r'\( \lim_{h\rightarrow +0}\frac{f(' + sympy.latex(_a) + r'+h)-f(' + sympy.latex(_a) + r')}{h} \)'
        quiz.answers.append(dict(ans))
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        [_var, _f, _a] = quiz.data
        answers = []
        ans = dict()
        ans['fraction'] = 100
        ans['data'] = r'\( \lim_{' + sympy.latex(_var) + r'\rightarrow' + sympy.latex(_a) + r'}\frac{f(' + sympy.latex(_var) + r')+f(' + sympy.latex(_a) + r')}{' + sympy.latex(_var)
        if _a < 0:
            ans['data'] += r'+' + sympy.latex(-_a) + r'} \)'
        elif _a == 0:
            ans['data'] += r'} \)'
        else:
            ans['data'] += r'-' + sympy.latex(_a) + r'} \)'
        answers.append(dict(ans))
        if _a != 0:
            ans['data'] = r'\( \lim_{h\rightarrow 0}\frac{f(' + sympy.latex(_a) + r'+h)+f(' + sympy.latex(_a) + r')}{h} \)'
        else:
            ans['data'] = r'\( \lim_{h\rightarrow 0}\frac{f(h)+f(' + sympy.latex(_a) + r')}{h} \)'
        answers.append(dict(ans))
        if _a != 0:
            ans['data'] = r'\( \lim_{' + sympy.latex(_var) + r'\rightarrow' + sympy.latex(_a) + r'-0}\frac{f(' + sympy.latex(_var) + r')+f(' + sympy.latex(_a) + r')}{' + sympy.latex(_var)
            if _a < 0:
                ans['data'] += r'+' + sympy.latex(-_a) + r'} \)'
            elif _a == 0:
                ans['data'] += r'} \)'
            else:
                ans['data'] += r'-' + sympy.latex(_a) + r'} \)'
            answers.append(dict(ans))
            ans['data'] = r'\( \lim_{' + sympy.latex(_var) + r'\rightarrow' + sympy.latex(_a) + r'+0}\frac{f(' + sympy.latex(_var) + r')+f(' + sympy.latex(_a) + r')}{' + sympy.latex(_var)
            if _a < 0:
                ans['data'] += r'+' + sympy.latex(-_a) + r'} \)'
            elif _a == 0:
                ans['data'] += r'} \)'
            else:
                ans['data'] += r'-' + sympy.latex(_a) + r'} \)'
            answers.append(dict(ans))
        else:
            ans['data'] = r'\( \lim_{' + sympy.latex(_var) + r'\rightarrow -0}\frac{f(' + sympy.latex(_var) + r')+f(' + sympy.latex(_a) + r')}{' + sympy.latex(_var) + r'} \)'
            answers.append(dict(ans))
            ans['data'] = r'\( \lim_{' + sympy.latex(_var) + r'\rightarrow +0}\frac{f(' + sympy.latex(_var) + r')+f(' + sympy.latex(_a) + r')}{' + sympy.latex(_var) + r'} \)'
            answers.append(dict(ans))
        ans['data'] = r'\( \lim_{h\rightarrow -0}\frac{f(' + sympy.latex(_a) + r'+h)+f(' + sympy.latex(_a) + r')}{h} \)'
        answers.append(dict(ans))
        ans['data'] = r'\( \lim_{h\rightarrow +0}\frac{f(' + sympy.latex(_a) + r'+h)+f(' + sympy.latex(_a) + r')}{h} \)'
        answers.append(dict(ans))
        ans['data'] = r'\( \lim_{' + sympy.latex(_var) + r'\rightarrow' + sympy.latex(_a) + r'}\frac{f(' + sympy.latex(_var) + r')-f(' + sympy.latex(_a) + r')}{' + sympy.latex(_var)
        if _a < 0:
            ans['data'] += sympy.latex(_a) + r'} \)'
            answers.append(dict(ans))
        elif _a > 0:
            ans['data'] += r'+' + sympy.latex(_a) + r'} \)'
            answers.append(dict(ans))            
        if _a != 0:
            ans['data'] = r'\( \lim_{h\rightarrow ' + sympy.latex(_a) + r'}\frac{f(' + sympy.latex(_a) + r'+h)-f(' + sympy.latex(_a) + r')}{h} \)'
            answers.append(dict(ans))
            ans['data'] = r'\( \lim_{' + sympy.latex(_var) + r'\rightarrow' + sympy.latex(_a) + r'-0}\frac{f(' + sympy.latex(_var) + r')-f(' + sympy.latex(_a) + r')}{' + sympy.latex(_var)
            if _a < 0:
                ans['data'] += sympy.latex(_a) + r'} \)'
            elif _a > 0:
                ans['data'] += r'+' + sympy.latex(_a) + r'} \)'
            answers.append(dict(ans))
            ans['data'] = r'\( \lim_{' + sympy.latex(_var) + r'\rightarrow' + sympy.latex(_a) + r'+0}\frac{f(' + sympy.latex(_var) + r')-f(' + sympy.latex(_a) + r')}{' + sympy.latex(_var)
            if _a < 0:
                ans['data'] += sympy.latex(_a) + r'} \)'
            elif _a > 0:
                ans['data'] += r'+' + sympy.latex(_a) + r'} \)'
            answers.append(dict(ans))
        return [random.choice(answers)]        
    def incorrect_answers_generate(self, quiz, size=4):
        # 個別には作らないので何もしない
        pass
    def question_text(self, quiz):
        [_var, _f, _a] = quiz.data
        _text = r'次の関数\( f(' + sympy.latex(_var) + r') \)の\( ' + sympy.latex(_var) + r'=' + sympy.latex(_a) + r' \)での微分係数が存在するか調べたり計算することに関係のない数式を選択してください。<br />'
        _text += r'\( f(' + sympy.latex(_var) +  r')=' 
        if isinstance(_f, list):
            _text += r'\left\{\begin{array}{cc}' 
            _text += sympy.latex(_f[0], order='lex') + r'&(' + sympy.latex(_var) + r'\geq ' + sympy.latex(_a) + r')\\'
            _text += sympy.latex(_f[1], order='lex') + r'&(' + sympy.latex(_var) + r'<' + sympy.latex(_a) + r')'
            _text += r'\end{array}\right.'
        else:
            _text += sympy.latex(_f, order='lex')
        _text += r' \)'
        return _text
    def answer_text(self, ans):
        return ans['data']


# In[101]:


if __name__ == "__main__":
    q = differential_coefficient_intermediate_expression()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=25)


# In[102]:


if __name__ == "__main__":
    pass
    #qz.save('differential_coefficient_intermediate_expression.xml')


# ## derivative by definition

# In[118]:


class derivative_by_definition(core.Question):
    name = '導関数の途中式（定義に基づく極限計算による）'
    _a = sympy.Symbol('a') # possibly zero
    _b = sympy.Symbol('b') # possibly zero
    _c = sympy.Symbol('c') # possibly zero
    _s = sympy.Symbol('s') # non-zero
    _t = sympy.Symbol('t') # non-zero
    _p = sympy.Symbol('p') # positive
    _e = sympy.Symbol('e') # non-one positive
    _x = sympy.Symbol('x')
    _h = sympy.Symbol('h')
    _skeletons = [_s*_x**2+_b*_x+_c, _s*_x+_c, (_s*_x+_a)/(_t*_x+_b), _s*_x**_p, 
                  _s*sympy.sin(_x), _s*sympy.cos(_x), _s*sympy.tan(_x), _s*sympy.exp(_x), 
                  _s*_e**_x, _s*sympy.log(_x)]
    def __init__(self, emin=-5, emax=5):
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
    def _function_generate(self, _func):
        _func = _func.subs(self._a, random.randint(self.elem_min, self.elem_max))
        _func = _func.subs(self._b, random.randint(self.elem_min, self.elem_max))
        _func = _func.subs(self._c, random.randint(self.elem_min, self.elem_max))
        _func = _func.subs(self._s, nonzero_randint(self.elem_min, self.elem_max))
        _func = _func.subs(self._t, nonzero_randint(self.elem_min, self.elem_max))
        _func = _func.subs(self._p, abs(nonzero_randint(self.elem_min, self.elem_max)))
        _func = _func.subs(self._e, nonone_randpint(self.elem_min, self.elem_max))
        return _func
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='導関数の途中式', quiz_number=_quiz_number)
        _func = random.choice(self._skeletons)
        _func = self._function_generate(_func)
        while _func.is_real:
            _func = random.choice(self._skeletons)
            _func = self._function_generate(_func)            
        quiz.quiz_identifier = hash(str(_func))
        # 正答の選択肢の生成
        _ime = sympy.simplify((_func.subs(self._x, self._x + self._h) - _func)/self._h)
        quiz.data = [_func, _ime]
        ans = { 'fraction': 100, 'data': _ime }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_func, _ime] = quiz.data
        _incorrect = sympy.simplify((_func.subs(self._x, self._x + self._h) - _func))
        ans['feedback'] = r'導関数を定義に基づき求める立式を行ってください。分母が欠けていることがわかります。'
        ans['data'] = _incorrect
        answers.append(dict(ans))
        _incorrect = sympy.simplify((_func.subs(self._x, self._x + self._h) + _func))
        ans['feedback'] = r'導関数を定義に基づき求める立式を行ってください。分母が欠けていることがわかります。'
        ans['data'] = _incorrect
        answers.append(dict(ans))
        _incorrect = sympy.simplify((_func.subs(self._x, self._x + self._h) + _func)/self._h)
        if sympy.simplify(_incorrect - _ime) != 0:
            ans['feedback'] = r'導関数を定義に基づき求める立式を行ってください。一部の符号に誤りがあることがわかります。'
            ans['data'] = _incorrect
            answers.append(dict(ans))
        _incorrect = sympy.simplify((_func.subs(self._x, self._x + self._h) - _func.subs(self._x, self._x - self._h))/self._h)
        if sympy.simplify(_incorrect - _ime) != 0:
            ans['feedback'] = r'導関数を定義に基づき求める立式を行ってください。一部の符号に誤りがあることがわかります。'
            ans['data'] = _incorrect
            answers.append(dict(ans))
        _incorrect = sympy.simplify((_func.subs(self._x, self._x + self._h) + _func.subs(self._x, self._x - self._h))/self._h)
        if sympy.simplify(_incorrect - _ime) != 0:
            ans['feedback'] = r'導関数を定義に基づき求める立式を行ってください。一部の符号に誤りがあることがわかります。'
            ans['data'] = _incorrect
            answers.append(dict(ans))
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_func, _ime] = quiz.data
        _text = r'次の関数\( f(x) \)の導関数を定義に基づき極限計算により求める際に，表れる途中式を選択してください。<br />'
        _text += r'\( f(x)=' + sympy.latex(_func, order='lex') + r' \)'
        return _text
    def answer_text(self, ans):
        return r'\( \lim_{' + sympy.latex(self._h) + r'\rightarrow 0}' +  sympy.latex(ans['data'], order='lex') + r' \)'


# In[119]:


if __name__ == "__main__":
    q = derivative_by_definition()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[120]:


if __name__ == "__main__":
    pass
    #qz.save('derivative_by_definition.xml')


# ## DifferentiableFunction class for the following quizzes

# In[40]:


class DifferentiableFunction():
    _const = sympy.Symbol('c') # non-zero integer
    _base = sympy.Symbol('a') # non-one positive integer
    _n = sympy.Symbol('n') # positive integer
    _x = sympy.Symbol('x')
    # in sympy, there is no way to represent log_a(x) without expanding.....
    # format: [f, df, incorrect dfs]
    _function_types = ['constant', 'monomial', 'sine', 'cosine', 'tangent', 
                       'natural_exponent', 'general_exponent', 'natural_logarithm']
    _function_defs = dict()
    _function_defs['constant'] = [_const, 0, [_const, 1]]
    _function_defs['monomial'] = [_x**_n, _n*_x**(_n-1), [_x**_n, _n*_x**_n, (_n-1)*_x**(_n-1)]]
    _function_defs['sine'] = [sympy.sin(_x), sympy.cos(_x), [sympy.sin(_x), -sympy.cos(_x)]]
    _function_defs['cosine'] = [sympy.cos(_x), -sympy.sin(_x), [sympy.cos(_x), sympy.sin(_x)]]
    _function_defs['tangent'] = [sympy.tan(_x), 1/sympy.cos(_x)**2, [sympy.tan(_x), 1/sympy.cos(_x), 1/sympy.sin(_x)**2]]
    _function_defs['natural_exponent'] = [sympy.exp(_x), sympy.exp(_x), [sympy.exp(_x-1)]]
    _function_defs['general_exponent'] = [_base**_x, sympy.log(_base)*_base**_x, [_base**_x, _base**(_x-1)]]
    _function_defs['natural_logarithm'] = [sympy.log(abs(_x)), 1/_x, [sympy.log(abs(_x))]]
    # func = logarithmic | linearity_all
    # logarithmic = linearity_mono**linearity_mono
    # linearity_all = [linearity_mono]+
    # linearity_mono = scalar*multi | scalar*quot | scalar*composite | scalar*elemental_func
    # composite|multi*quot = linearity @@ linearity
    # linearity = elemental_func | elemental_func + constant
    _expression_types = ['exponential', 'summation', 'scalar', 'composition', 'multiplication', 'quotient']
    _incorrect_reasons = ['formula', 'sign', 'composition', 'scalar', 'part']
    def __init__(self, emin=-3, emax=3, nmin=1, nmax=3, lrate=0, lmin=1, lmax=2, crate=0.5, mrate=0.3, qrate=0.3, srate=0.25, inclog=True):
        # 生成する個々の係数等の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 生成する次数の範囲
        self.deg_min = nmin
        self.deg_max = nmax
        # 対数微分法を最後に付与する確率
        self.logarithmic_ratio = lrate
        # 線形性に表れる関数の個数
        self.linearity_min = lmin
        self.linearity_max = lmax
        # それぞれの合成関数の確率
        self.composite_ratio = crate
        # それぞれの積の微分法の確率
        self.multiplication_ratio = mrate
        # それぞれの商の微分法の確率
        self.quotient_ratio = qrate
        # 定数項の確率
        self.constant_term_ratio = srate
        # 対数関数を選択肢に入れるか
        if not inclog:
            self._function_types = self._function_types[:-1]
        # internals
        self.function = 0
    # func = logarithmic | linearity_all
    # logarithmic = linearity_mono**linearity_mono
    def generate_function(self):
        if random.random() < self.logarithmic_ratio:
            _funcE = self._generate_linearity_mono()
            _funcB = self._generate_linearity_mono()
            _func = ['exponential', _funcB, _funcE]
        else:
            _func = self._generate_linearity_all()
        self.function = _func
    # linearity_all = [linearity_mono]+
    def _generate_linearity_all(self):
        _n = random.randint(self.linearity_min, self.linearity_max)
        _func1 = self._generate_linearity_mono()
        if _n == 1:
            return _func1
        _func = ['summation', _func1]
        for _i in range(_n-1):
            _func.append(self._generate_linearity_mono())
        return _func
    # linearity_mono = scalar*multi | scalar*quot | scalar*composite | scalar*elemental_func
    def _generate_linearity_mono(self):
        _scalar = nonzero_randint(self.elem_min, self.elem_max)
        if random.random() < self.multiplication_ratio:
            _types = random.sample(self._function_types[1:], 2)
            _func = ['scalar', _scalar, self._generate_multiplication(_types)]
        elif random.random() < self.quotient_ratio:
            _types = random.sample(self._function_types[1:], 2)
            _func = ['scalar', _scalar, self._generate_quotient(_types)]
        elif random.random() < self.composite_ratio:
            _types = random.sample(self._function_types[1:], 2)
            _func = ['scalar', _scalar, self._generate_composition(_types)]
        else:
            _func = ['scalar', _scalar, self._generate_elemental_func()]
        return _func
    # composite|multi*quot = linearity @@ linearity
    def _generate_composition(self, _types):
        _func1 = self._generate_linearity(_types[0])
        _func2 = self._generate_linearity(_types[1])
        return ['composition', _func1, _func2]
    def _generate_multiplication(self, _types):
        _func1 = self._generate_linearity(_types[0])
        _func2 = self._generate_linearity(_types[1])
        return ['multiplication', _func1, _func2]
    def _generate_quotient(self, _types):
        _func1 = self._generate_linearity(_types[0])
        _func2 = self._generate_linearity(_types[1])
        return ['quotient', _func1, _func2]
    # linearity = elemental_func | elemental_func + constant
    def _generate_linearity(self, _type):
        _scalar = nonzero_randint(self.elem_min, self.elem_max)
        _func1 = ['scalar', _scalar, self._generate_elemental_func(_type)]
        if random.random() >= self.constant_term_ratio:
            return _func1
        else:
            _func = ['summation', _func1]
            _func.append(self._generate_elemental_func('constant'))
        return _func
    def _mysubs(self, target, var, repl):
        if sympy.sympify(target).is_real:
            return target
        elif len(var) == 1:
            return target.subs(var[0], repl[0])
        else:
            return self._mysubs(target.subs(var[0], repl[0]), var[1:], repl[1:])
    def _generate_elemental_func(self, _type=None):
        if _type is None:
            _type = random.choice(self._function_types[1:])
        _vars = [self._const, self._base, self._n]
        _repl = [nonzero_randint(self.elem_min, self.elem_max), 
                 nonone_randpint(self.elem_min, self.elem_max),
                 abs(nonzero_randint(self.deg_min, self.deg_max))]
        _func = self._function_defs[_type]
        _func = [self._mysubs(_func[0], _vars, _repl), 
                 self._mysubs(_func[1], _vars, _repl), 
                [self._mysubs(_f, _vars, _repl) for _f in _func[2]]]
        return _func
    def _get_function(self, _func):
        _recursive_call = self._get_function
        # ['exponential', 'summation', 'scalar', 'composition', 'multiplication', 'quotient']  
        if _func[0] == 'exponential':
            return _recursive_call(_func[1]) ** _recursive_call(_func[2])
        elif _func[0] == 'summation':
            _summand = 0
            for _f in _func[1:]:
                _summand = _summand + _recursive_call(_f)
            return _summand
        elif _func[0] == 'scalar':
            return _func[1] * _recursive_call(_func[2])
        elif _func[0] == 'composition':
            return _recursive_call(_func[1]).subs(self._x, _recursive_call(_func[2]))
        elif _func[0] == 'multiplication':
            return _recursive_call(_func[1]) * _recursive_call(_func[2])
        elif _func[0] == 'quotient':
            return _recursive_call(_func[1]) / _recursive_call(_func[2])
        else:
            return _func[0]
    def get_function(self):
        if self.function == 0:
            self.generate_function()
        return self._get_function(self.function)
    def _get_derivative(self, _func):
        _recursive_call = self._get_derivative
        # ['exponential', 'summation', 'scalar', 'composition', 'multiplication', 'quotient']  
        if _func[0] == 'exponential':
            _rest = ['multiplication', _func[2], ['composition', [sympy.log(self._x), 1/self._x, [sympy.log(self._x)]], _func[1]]]
            return (self._get_function(_func[1]) ** self._get_function(_func[2])) * _recursive_call(_rest)
        elif _func[0] == 'summation':
            _summand = 0
            for _f in _func[1:]:
                _summand = _summand + _recursive_call(_f)
            return _summand
        elif _func[0] == 'scalar':
            return _func[1] * _recursive_call(_func[2])
        elif _func[0] == 'composition':
            return _recursive_call(_func[2]) * self._mysubs(_recursive_call(_func[1]), [self._x], [self._get_function(_func[2])])
        elif _func[0] == 'multiplication':
            return _recursive_call(_func[1]) * self._get_function(_func[2]) + self._get_function(_func[1]) * _recursive_call(_func[2])
        elif _func[0] == 'quotient':
            return (_recursive_call(_func[1]) * self._get_function(_func[2]) - self._get_function(_func[1]) * _recursive_call(_func[2]))/(self._get_function(_func[2])**2)
        else:
            return _func[1]
    def get_derivative(self):
        if self.function == 0:
            self.generate_function()
#        display(sympy.diff(self.get_function(),self._x))
#        display(sympy.simplify(self._get_derivative(self.function) - sympy.diff(self.get_function(),self._x)))
        return self._get_derivative(self.function)
    def get_higher_derivative(self, n=1):
        if self.function == 0:
            self.generate_function()
        if n <= 0:
            return self.get_function()
        else:
            _df = self.get_derivative()
            return sympy.diff(_df, self._x, n-1)        
    def _get_incorrect_derivatives_by_formula(self, _func):
        _recursive_call = self._get_incorrect_derivatives_by_formula
        # ['exponential', 'summation', 'scalar', 'composition', 'multiplication', 'quotient']  
        if _func[0] == 'exponential':
            _rest = ['multiplication', _func[2], ['composition', [sympy.log(self._x), 1/self._x, [sympy.log(self._x)]], _func[1]]]
            return [(self._get_function(_func[1]) ** self._get_function(_func[2])) * _f for _f in _recursive_call(_rest)]
        elif _func[0] == 'summation':
            _summand = [0]
            for _f in _func[1:]:
                _dfs = _recursive_call(_f)
                _new_summand = []
                for _df in _dfs:
                    _new_summand = _new_summand + [ _s + _df for _s in _summand]
                _summand = _new_summand
            return _summand
        elif _func[0] == 'scalar':
            return [_func[1] * _f for _f in _recursive_call(_func[2])]
        elif _func[0] == 'composition':
            _funcs = []
            for _df1 in _recursive_call(_func[1]):
                for _df2 in _recursive_call(_func[2]):
                    _funcs.append(_df2 * self._mysubs(_df1, [self._x], [self._get_function(_func[2])]))
            return _funcs
        elif _func[0] == 'multiplication':
            _funcs = []
            for _df1 in _recursive_call(_func[1]):
                for _df2 in _recursive_call(_func[2]):
                    _funcs.append(_df1 * self._get_function(_func[2]) + self._get_function(_func[1]) * _df2)
            return _funcs
        elif _func[0] == 'quotient':
            _funcs = []
            for _df1 in _recursive_call(_func[1]):
                for _df2 in _recursive_call(_func[2]):
                    _funcs.append((_df1 * self._get_function(_func[2]) - self._get_function(_func[1]) * _df2)/(self._get_function(_func[2])**2))
            return _funcs
        else:
            return _func[2]
    def _get_incorrect_derivatives_by_sign(self, _func):
        _recursive_call = self._get_incorrect_derivatives_by_sign
        # ['exponential', 'summation', 'scalar', 'composition', 'multiplication', 'quotient']  
        if _func[0] == 'exponential':
            _rest = ['multiplication', _func[2], ['composition', [sympy.log(self._x), 1/self._x, [sympy.log(self._x)]], _func[1]]]
            return [(self._get_function(_func[1]) ** self._get_function(_func[2])) * _f for _f in _recursive_call(_rest)]
        elif _func[0] == 'summation':
            _summand = [0]
            for _f in _func[1:]:
                _dfs = _recursive_call(_f)
                _new_summand = []
                for _df in _dfs:
                    _new_summand = _new_summand + [ _s + _df for _s in _summand]
                _summand = _new_summand
            return _summand
        elif _func[0] == 'scalar':
            return [_func[1] * _f for _f in _recursive_call(_func[2])]
        elif _func[0] == 'composition':
            _funcs = []
            for _df1 in _recursive_call(_func[1]):
                for _df2 in _recursive_call(_func[2]):
                    _funcs.append(_df2 * self._mysubs(_df1, [self._x], [self._get_function(_func[2])]))
            return _funcs
        elif _func[0] == 'multiplication':
            _funcs = []
            for _df1 in _recursive_call(_func[1]):
                for _df2 in _recursive_call(_func[2]):
                    _funcs.append(_df1 * self._get_function(_func[2]) - self._get_function(_func[1]) * _df2)
            return _funcs
        elif _func[0] == 'quotient':
            _funcs = []
            for _df1 in _recursive_call(_func[1]):
                for _df2 in _recursive_call(_func[2]):
                    _funcs.append((_df1 * self._get_function(_func[2]) + self._get_function(_func[1]) * _df2)/(self._get_function(_func[2])**2))
            return _funcs
        else:
            return [_func[1]]
    def _get_incorrect_derivatives_by_composition(self, _func):
        _recursive_call = self._get_incorrect_derivatives_by_composition
        # ['exponential', 'summation', 'scalar', 'composition', 'multiplication', 'quotient']  
        if _func[0] == 'exponential':
            _rest = ['multiplication', _func[2], ['composition', [sympy.log(self._x), 1/self._x, [sympy.log(self._x)]], _func[1]]]
            return [(self._get_function(_func[1]) ** self._get_function(_func[2])) * _f for _f in _recursive_call(_rest)]
        elif _func[0] == 'summation':
            _summand = [0]
            for _f in _func[1:]:
                _dfs = _recursive_call(_f)
                _new_summand = []
                for _df in _dfs:
                    _new_summand = _new_summand + [ _s + _df for _s in _summand]
                _summand = _new_summand
            return _summand
        elif _func[0] == 'scalar':
            return [_func[1] * _f for _f in _recursive_call(_func[2])]
        elif _func[0] == 'composition':
            _funcs = []
            for _df1 in _recursive_call(_func[1]):
                _funcs.append(_df1 )
                for _df2 in _recursive_call(_func[2]):
                    _funcs.append(_df2 )
                    _funcs.append(_df1 * self._mysubs(_df2, [self._x], [self._get_function(_func[2])]))
            return _funcs
        elif _func[0] == 'multiplication':
            _funcs = []
            for _df1 in _recursive_call(_func[1]):
                for _df2 in _recursive_call(_func[2]):
                    _funcs.append(_df1 * self._get_function(_func[2]) + self._get_function(_func[1]) * _df2)
            return _funcs
        elif _func[0] == 'quotient':
            _funcs = []
            for _df1 in _recursive_call(_func[1]):
                for _df2 in _recursive_call(_func[2]):
                    _funcs.append((_df1 * self._get_function(_func[2]) - self._get_function(_func[1]) * _df2)/(self._get_function(_func[2])**2))
            return _funcs
        else:
            return [_func[1]]
    def _get_incorrect_derivatives_by_scalar(self, _func):
        _recursive_call = self._get_incorrect_derivatives_by_scalar
        # ['exponential', 'summation', 'scalar', 'composition', 'multiplication', 'quotient']  
        if _func[0] == 'exponential':
            _rest = ['multiplication', _func[2], ['composition', [sympy.log(self._x), 1/self._x, [sympy.log(self._x)]], _func[1]]]
            return [(self._get_function(_func[1]) ** self._get_function(_func[2])) * _f for _f in _recursive_call(_rest)]
        elif _func[0] == 'summation':
            _summand = [0]
            for _f in _func[1:]:
                _dfs = _recursive_call(_f)
                _new_summand = []
                for _df in _dfs:
                    _new_summand = _new_summand + [ _s + _df for _s in _summand]
                _summand = _new_summand
            return _summand
        elif _func[0] == 'scalar':
            return [_f for _f in _recursive_call(_func[2])] + [-_f for _f in _recursive_call(_func[2])]
        elif _func[0] == 'composition':
            _funcs = []
            for _df1 in _recursive_call(_func[1]):
                for _df2 in _recursive_call(_func[2]):
                    _funcs.append(_df2 * self._mysubs(_df1, [self._x], [self._get_function(_func[2])]))
            return _funcs
        elif _func[0] == 'multiplication':
            _funcs = []
            for _df1 in _recursive_call(_func[1]):
                for _df2 in _recursive_call(_func[2]):
                    _funcs.append(_df1 * self._get_function(_func[2]) + self._get_function(_func[1]) * _df2)
            return _funcs
        elif _func[0] == 'quotient':
            _funcs = []
            for _df1 in _recursive_call(_func[1]):
                for _df2 in _recursive_call(_func[2]):
                    _funcs.append((_df1 * self._get_function(_func[2]) - self._get_function(_func[1]) * _df2)/(self._get_function(_func[2])**2))
            return _funcs
        else:
            return [_func[1]]
    def _get_incorrect_derivatives_by_part(self, _func):
        _recursive_call = self._get_incorrect_derivatives_by_part
        # ['exponential', 'summation', 'scalar', 'composition', 'multiplication', 'quotient']  
        if _func[0] == 'exponential':
            _rest = ['multiplication', _func[2], ['composition', [sympy.log(self._x), 1/self._x, [sympy.log(self._x)]], _func[1]]]
            return [(self._get_function(_func[1]) ** self._get_function(_func[2])) * _f for _f in _recursive_call(_rest)]
        elif _func[0] == 'summation':
            _funcs = []
            for _f in _func[1:]:
                _funcs = _funcs + _recursive_call(_f)
            return _funcs
        elif _func[0] == 'scalar':
            return [_func[1] * _f for _f in _recursive_call(_func[2])]
        elif _func[0] == 'composition':
            _funcs = []
            for _df1 in _recursive_call(_func[1]):
                for _df2 in _recursive_call(_func[2]):
                    _funcs.append(_df2 * self._mysubs(_df1, [self._x], [self._get_function(_func[2])]))
            return _funcs
        elif _func[0] == 'multiplication':
            _funcs = []
            for _df1 in _recursive_call(_func[1]):
                for _df2 in _recursive_call(_func[2]):
                    _funcs.append(_df1 * self._get_function(_func[2]) + self._get_function(_func[1]) * _df2)
            return _funcs
        elif _func[0] == 'quotient':
            _funcs = []
            for _df1 in _recursive_call(_func[1]):
                for _df2 in _recursive_call(_func[2]):
                    _funcs.append((_df1 * self._get_function(_func[2]) - self._get_function(_func[1]) * _df2)/(self._get_function(_func[2])**2))
            return _funcs
        else:
            return [_func[1]]
    def _not_same_check(self, value):
        if not value.is_real:
            return False
        if value > 1.0e-4:
            return True
        return False
    def _incorrect_derivarives_only(self, _dfs):
        _funcs = []
        _correct_df = self.get_derivative()
        _correct_df0 = self._mysubs(_correct_df, [self._x], [0])
        _correct_df1 = self._mysubs(_correct_df, [self._x], [1])
        _correct_df2 = self._mysubs(_correct_df, [self._x], [2])
        for _df in _dfs:
            if _df in _funcs:
                continue
            if self._not_same_check(abs(sympy.N(_correct_df0 - self._mysubs(_df, [self._x], [0])))):
                _funcs.append(_df)
            elif self._not_same_check(abs(sympy.N(_correct_df1 - self._mysubs(_df, [self._x], [1])))):
                _funcs.append(_df)
            elif self._not_same_check(abs(sympy.N(_correct_df2 - self._mysubs(_df, [self._x], [2])))):
                _funcs.append(_df)
        return _funcs
    def get_incorrect_derivatives(self, _type=None):
        if self.function == 0:
            self.generate_function()
        if _type is None:
            _type = random.choice(self._incorrect_reasons)
        if _type == 'formula':
            _dfs = self._get_incorrect_derivatives_by_formula(self.function)
        elif _type == 'sign':
            _dfs = self._get_incorrect_derivatives_by_sign(self.function)
        elif _type == 'composition':
            _dfs = self._get_incorrect_derivatives_by_composition(self.function)
        elif _type == 'scalar':
            _dfs = self._get_incorrect_derivatives_by_scalar(self.function)
        elif _type == 'part':
            _dfs = self._get_incorrect_derivatives_by_part(self.function)
        return self._incorrect_derivarives_only(_dfs)
    def _incorrect_higher_derivarives_only(self, _dfs, n):
        _funcs = []
        _correct_df = self.get_higher_derivative(n)
        _correct_df0 = self._mysubs(_correct_df, [self._x], [0])
        _correct_df1 = self._mysubs(_correct_df, [self._x], [1])
        _correct_df2 = self._mysubs(_correct_df, [self._x], [2])
        for _df in _dfs:
            if _df in _funcs:
                continue
            if self._not_same_check(abs(sympy.N(_correct_df0 - self._mysubs(_df, [self._x], [0])))):
                _funcs.append(_df)
            elif self._not_same_check(abs(sympy.N(_correct_df1 - self._mysubs(_df, [self._x], [1])))):
                _funcs.append(_df)
            elif self._not_same_check(abs(sympy.N(_correct_df2 - self._mysubs(_df, [self._x], [2])))):
                _funcs.append(_df)
        return _funcs
    def get_incorrect_higher_derivatives(self, n=1, _type=None):
        if self.function == 0:
            self.generate_function()
        if _type is None:
            _type = random.choice(self._incorrect_reasons + ['degree'])
        if _type == 'formula':
            _dfs = self._get_incorrect_derivatives_by_formula(self.function)
        elif _type == 'sign':
            _dfs = self._get_incorrect_derivatives_by_sign(self.function)
        elif _type == 'composition':
            _dfs = self._get_incorrect_derivatives_by_composition(self.function)
        elif _type == 'scalar':
            _dfs = self._get_incorrect_derivatives_by_scalar(self.function)
        elif _type == 'part':
            _dfs = self._get_incorrect_derivatives_by_part(self.function)
        elif _type == 'degree':
            _dfs = [self.get_higher_derivative(_n) for _n in list(set(range(0,n+2))-set([n]))]
        return self._incorrect_higher_derivarives_only(_dfs, n)


# In[41]:


if __name__ == "__main__":
    df = DifferentiableFunction()
    df.generate_function()
    display(df.get_function())
    display(df.get_derivative())
    display(df.get_higher_derivative(2))
    display(df.get_incorrect_higher_derivatives(2, 'degree'))


# ## derivative by linearity

# In[312]:


class derivative_by_linearity(core.Question):
    name = '線形性に基づく導関数の計算（基本公式と線形性の活用）'
    def __init__(self, emin=-3, emax=3, nmin=1, nmax=3, lmin=2, lmax=3):
        # 生成する個々の係数等の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 生成する次数の範囲
        self.deg_min = nmin
        self.deg_max = nmax
        # 生成する関数の和の範囲
        self.func_min = lmin
        self.func_max = lmax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='線形性に基づく導関数の計算', quiz_number=_quiz_number)
        df = DifferentiableFunction(emin=self.elem_min, emax=self.elem_max, nmin=self.deg_min, nmax=self.deg_max, 
                                   lrate=0, lmin=self.func_min, lmax=self.func_max, crate=0, mrate=0, qrate=0, srate=0)
        _func = df.get_function()
        quiz.quiz_identifier = hash(str(_func))
        # 正答の選択肢の生成
        _df = df.get_derivative()
        quiz.data = [_func, _df, df]
        ans = { 'fraction': 100, 'data': _df }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_func, _df, df] = quiz.data
        ans['feedback'] = r'基本的な関数の導関数の公式を確認してください。'
        for _incorrect in df.get_incorrect_derivatives('formula'):
            ans['data'] = _incorrect
            answers.append(dict(ans))
        ans['feedback'] = r'導関数の線形性に基づいて計算してください。定数倍や加減算を忘れている可能性があります。'
        for _incorrect in df.get_incorrect_derivatives('scalar'):
            ans['data'] = _incorrect
            answers.append(dict(ans))
        for _incorrect in df.get_incorrect_derivatives('part'):
            ans['data'] = _incorrect
            answers.append(dict(ans))
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_func, _df, df] = quiz.data
        _text = r'次の関数\( f(x) \)の導関数を選択してください。'
        _text += r'<br />\( f(x)=' + sympy.latex(_func, order='lex') + r' \)'
        return _text
    def answer_text(self, ans):
        return r'\( ' +  sympy.latex(ans['data'], order='lex') + r' \)'


# In[313]:


if __name__ == "__main__":
    q = derivative_by_linearity()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=25)


# In[314]:


if __name__ == "__main__":
    pass
    #qz.save('derivative_by_linearity.xml')


# ## derivative by multiplication

# In[318]:


class derivative_by_multiplication(core.Question):
    name = '積の微分法に基づく導関数の計算（基本公式と積の微分法の活用）'
    def __init__(self, emin=-3, emax=3, nmin=1, nmax=3, lmin=1, lmax=1):
        # 生成する個々の係数等の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 生成する次数の範囲
        self.deg_min = nmin
        self.deg_max = nmax
        # 生成する関数の和の範囲
        self.func_min = lmin
        self.func_max = lmax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='積の微分法に基づく導関数の計算', quiz_number=_quiz_number)
        df = DifferentiableFunction(emin=self.elem_min, emax=self.elem_max, nmin=self.deg_min, nmax=self.deg_max, 
                                   lrate=0, lmin=self.func_min, lmax=self.func_max, crate=0, mrate=1, qrate=0, srate=0)
        _func = df.get_function()
        quiz.quiz_identifier = hash(str(_func))
        # 正答の選択肢の生成
        _df = df.get_derivative()
        quiz.data = [_func, _df, df]
        ans = { 'fraction': 100, 'data': _df }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_func, _df, df] = quiz.data
        ans['feedback'] = r'基本的な関数の導関数の公式を確認してください。'
        for _incorrect in df.get_incorrect_derivatives('formula'):
            ans['data'] = _incorrect
            answers.append(dict(ans))
        ans['feedback'] = r'導関数の線形性に基づいて計算してください。加減算を忘れている可能性があります。'
        for _incorrect in df.get_incorrect_derivatives('part'):
            ans['data'] = _incorrect
            answers.append(dict(ans))
        ans['feedback'] = r'積の微分法を確認しましょう。符号を間違えている可能性があります。'
        for _incorrect in df.get_incorrect_derivatives('sign'):
            ans['data'] = _incorrect
            answers.append(dict(ans))
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_func, _df, df] = quiz.data
        _text = r'次の関数\( f(x) \)の導関数を選択してください。'
        _text += r'<br />\( f(x)=' + sympy.latex(_func, order='lex') + r' \)'
        return _text
    def answer_text(self, ans):
        return r'\( ' +  sympy.latex(ans['data'], order='lex') + r' \)'


# In[319]:


if __name__ == "__main__":
    q = derivative_by_multiplication()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=25)


# In[320]:


if __name__ == "__main__":
    pass
    #qz.save('derivative_by_multiplication.xml')


# ## derivative by quotient

# In[321]:


class derivative_by_quotient(core.Question):
    name = '商の微分法に基づく導関数の計算（基本公式と商の微分法の活用）'
    def __init__(self, emin=-3, emax=3, nmin=1, nmax=3, lmin=1, lmax=1):
        # 生成する個々の係数等の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 生成する次数の範囲
        self.deg_min = nmin
        self.deg_max = nmax
        # 生成する関数の和の範囲
        self.func_min = lmin
        self.func_max = lmax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='商の微分法に基づく導関数の計算', quiz_number=_quiz_number)
        df = DifferentiableFunction(emin=self.elem_min, emax=self.elem_max, nmin=self.deg_min, nmax=self.deg_max, 
                                   lrate=0, lmin=self.func_min, lmax=self.func_max, crate=0, mrate=0, qrate=1, srate=0)
        _func = df.get_function()
        quiz.quiz_identifier = hash(str(_func))
        # 正答の選択肢の生成
        _df = df.get_derivative()
        quiz.data = [_func, _df, df]
        ans = { 'fraction': 100, 'data': _df }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_func, _df, df] = quiz.data
        ans['feedback'] = r'基本的な関数の導関数の公式を確認してください。'
        for _incorrect in df.get_incorrect_derivatives('formula'):
            ans['data'] = _incorrect
            answers.append(dict(ans))
        ans['feedback'] = r'導関数の線形性に基づいて計算してください。加減算を忘れている可能性があります。'
        for _incorrect in df.get_incorrect_derivatives('part'):
            ans['data'] = _incorrect
            answers.append(dict(ans))
        ans['feedback'] = r'商の微分法を確認しましょう。符号を間違えている可能性があります。'
        for _incorrect in df.get_incorrect_derivatives('sign'):
            ans['data'] = _incorrect
            answers.append(dict(ans))
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_func, _df, df] = quiz.data
        _text = r'次の関数\( f(x) \)の導関数を選択してください。'
        _text += r'<br />\( f(x)=' + sympy.latex(_func, order='lex') + r' \)'
        return _text
    def answer_text(self, ans):
        return r'\( ' +  sympy.latex(ans['data'], order='lex') + r' \)'


# In[322]:


if __name__ == "__main__":
    q = derivative_by_quotient()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=25)


# In[323]:


if __name__ == "__main__":
    pass
    #qz.save('derivative_by_quotient.xml')


# ## derivative by composition

# In[326]:


class derivative_by_composition(core.Question):
    name = '合成関数の微分法に基づく導関数の計算（基本公式と合成関数の微分法の活用）'
    def __init__(self, emin=-3, emax=3, nmin=2, nmax=3, lmin=1, lmax=1):
        # 生成する個々の係数等の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 生成する次数の範囲
        self.deg_min = nmin
        self.deg_max = nmax
        # 生成する関数の和の範囲
        self.func_min = lmin
        self.func_max = lmax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='合成関数の微分法に基づく導関数の計算', quiz_number=_quiz_number)
        df = DifferentiableFunction(emin=self.elem_min, emax=self.elem_max, nmin=self.deg_min, nmax=self.deg_max, 
                                   lrate=0, lmin=self.func_min, lmax=self.func_max, crate=1, mrate=0, qrate=0, srate=0)
        _func = df.get_function()
        quiz.quiz_identifier = hash(str(_func))
        # 正答の選択肢の生成
        _df = df.get_derivative()
        quiz.data = [_func, _df, df]
        ans = { 'fraction': 100, 'data': _df }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_func, _df, df] = quiz.data
        ans['feedback'] = r'基本的な関数の導関数の公式を確認してください。'
        for _incorrect in df.get_incorrect_derivatives('formula'):
            ans['data'] = _incorrect
            answers.append(dict(ans))
        ans['feedback'] = r'導関数の線形性に基づいて計算してください。加減算を忘れている可能性があります。'
        for _incorrect in df.get_incorrect_derivatives('part'):
            ans['data'] = _incorrect
            answers.append(dict(ans))
        ans['feedback'] = r'合成関数の微分法を確認しましょう。取り違えや取りこぼしがあります。'
        for _incorrect in df.get_incorrect_derivatives('composition'):
            ans['data'] = _incorrect
            answers.append(dict(ans))
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_func, _df, df] = quiz.data
        _text = r'次の関数\( f(x) \)の導関数を選択してください。'
        _text += r'<br />\( f(x)=' + sympy.latex(_func, order='lex') + r' \)'
        return _text
    def answer_text(self, ans):
        return r'\( ' +  sympy.latex(ans['data'], order='lex') + r' \)'


# In[327]:


if __name__ == "__main__":
    q = derivative_by_composition()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=25)


# In[328]:


if __name__ == "__main__":
    pass
    #qz.save('derivative_by_composition.xml')


# ## derivative by taking logarithm

# In[335]:


class derivative_by_taking_logarithm(core.Question):
    name = '対数微分法に基づく導関数の計算（基本公式と対数微分法の活用）'
    def __init__(self, emin=-3, emax=3, nmin=2, nmax=3, lmin=1, lmax=1):
        # 生成する個々の係数等の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 生成する次数の範囲
        self.deg_min = nmin
        self.deg_max = nmax
        # 生成する関数の和の範囲
        self.func_min = lmin
        self.func_max = lmax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='対数微分法に基づく導関数の計算', quiz_number=_quiz_number)
        df = DifferentiableFunction(emin=self.elem_min, emax=self.elem_max, nmin=self.deg_min, nmax=self.deg_max, 
                                   lrate=1, lmin=self.func_min, lmax=self.func_max, crate=0, mrate=0, qrate=0, srate=0)
        _func = df.get_function()
        quiz.quiz_identifier = hash(str(_func))
        # 正答の選択肢の生成
        _df = df.get_derivative()
        quiz.data = [_func, _df, df]
        ans = { 'fraction': 100, 'data': _df }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_func, _df, df] = quiz.data
        ans['feedback'] = r'基本的な関数の導関数の公式を確認してください。'
        for _incorrect in df.get_incorrect_derivatives('formula'):
            ans['data'] = _incorrect
            answers.append(dict(ans))
        ans['feedback'] = r'対数微分法の途中の計算で部分的に間違えていないでしょうか。'
        for _incorrect in df.get_incorrect_derivatives('sign'):
            ans['data'] = _incorrect
            answers.append(dict(ans))
        for _incorrect in df.get_incorrect_derivatives('composition'):
            ans['data'] = _incorrect
            answers.append(dict(ans))
        for _incorrect in df.get_incorrect_derivatives('part'):
            ans['data'] = _incorrect
            answers.append(dict(ans))
        for _incorrect in df.get_incorrect_derivatives('scalar'):
            ans['data'] = _incorrect
            answers.append(dict(ans))
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_func, _df, df] = quiz.data
        _text = r'次の関数\( f(x) \)の導関数を選択してください。'
        _text += r'<br />\( f(x)=' + sympy.latex(_func, order='lex') + r' \)'
        return _text
    def answer_text(self, ans):
        return r'\( ' +  sympy.latex(ans['data'], order='lex') + r' \)'


# In[336]:


if __name__ == "__main__":
    q = derivative_by_taking_logarithm()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=25)


# In[337]:


if __name__ == "__main__":
    pass
    #qz.save('derivative_by_taking_logarithm.xml')


# ## higher derivative

# In[53]:


class higher_derivative(core.Question):
    name = '高階導関数の計算'
    def __init__(self, emin=-4, emax=4, nmin=2, nmax=3, lmin=1, lmax=1, dmin=2, dmax=3):
        # 生成する個々の係数等の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 生成する次数の範囲
        self.deg_min = nmin
        self.deg_max = nmax
        # 生成する関数の和の範囲
        self.func_min = lmin
        self.func_max = lmax
        # 何階の導関数を求めるか
        self.order_min = dmin
        self.order_max = dmax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='高階導関数の計算', quiz_number=_quiz_number)
        _order = random.randint(self.order_min, self.order_max)
        if _order <= 2:
            df = DifferentiableFunction(emin=self.elem_min, emax=self.elem_max, nmin=self.deg_min, nmax=self.deg_max, inclog=False,
                                   lrate=0, lmin=self.func_min, lmax=self.func_max, crate=0.25, mrate=0.25, qrate=0, srate=0.25)
        else:
            df = DifferentiableFunction(emin=self.elem_min, emax=self.elem_max, nmin=self.deg_min, nmax=self.deg_max, inclog=False,
                                   lrate=0, lmin=self.func_min, lmax=self.func_max, crate=0, mrate=0, qrate=0, srate=0.25)
        _func = df.get_function()
        quiz.quiz_identifier = hash(str(_func) + str(_order))
        # 正答の選択肢の生成
        _df = df.get_higher_derivative(_order)
        quiz.data = [_func, _order, _df, df]
        ans = { 'fraction': 100, 'data': _df }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_func, _order, _df, df] = quiz.data
        ans['feedback'] = r'高階導関数とは，階数分の微分を行った関数です。'
        for _incorrect in df.get_incorrect_higher_derivatives(_order, 'degree'):
            ans['data'] = _incorrect
            answers.append(dict(ans))
        answers = common.answer_union(answers)
        additional_answers = []
        ans['feedback'] = r'それぞれの微分について，計算を確認してください。'
        for _incorrect in df.get_incorrect_higher_derivatives(_order, 'formula'):
            ans['data'] = _incorrect
            additional_answers.append(dict(ans))
        for _incorrect in df.get_incorrect_higher_derivatives(_order, 'sign'):
            ans['data'] = _incorrect
            additional_answers.append(dict(ans))
        for _incorrect in df.get_incorrect_higher_derivatives(_order, 'composition'):
            ans['data'] = _incorrect
            additional_answers.append(dict(ans))
        for _incorrect in df.get_incorrect_higher_derivatives(_order, 'part'):
            ans['data'] = _incorrect
            additional_answers.append(dict(ans))
        for _incorrect in df.get_incorrect_higher_derivatives(_order, 'scalar'):
            ans['data'] = _incorrect
            additional_answers.append(dict(ans))
        additional_answers = common.answer_union(additional_answers)
        if len(additional_answers) > len(answers):
            additional_answers = random.sample(additional_answers, k=max(len(answers),min(size-len(answers),len(additional_answers))))
        answers = answers + additional_answers
        answers = common.answer_union(answers)
        if len(answers) > size:
            answers = random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_func, _order, _df, df] = quiz.data
        _text = r'次の関数\( f(x) \)の' + str(_order) + r'階導関数を選択してください。'
        _text += r'<br />\( f(x)=' + sympy.latex(_func, order='lex') + r' \)'
        return _text
    def answer_text(self, ans):
        return r'\( ' +  sympy.latex(ans['data'], order='lex') + r' \)'


# In[54]:


if __name__ == "__main__":
    q = higher_derivative()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=25)


# In[55]:


if __name__ == "__main__":
    pass
    #qz.save('higher_derivative.xml')


# ## local minimum and maximum

# In[124]:


class local_minimum_maximum(core.Question):
    name = '極大値と極小値（増減表からの決定）'    
    def __init__(self, emax=5, nmin=1, nmax=3):
        # 生成するdf=0の個数範囲
        self.nsp_min = nmin
        self.nsp_max = nmax
        # 数の生成
        self.numbers = list(range(0,emax+1)) + [sympy.sqrt(sympy.prime(i)) for i in range(1,emax)] + [sympy.pi, sympy.E]
        self.numbers = self.numbers + [_e/sympy.Integer(2) for _e in self.numbers]
        self.numbers = self.numbers + [-_e for _e in self.numbers]
        self.numbers = sorted(list(set(self.numbers)))
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='極大値と極小値', quiz_number=_quiz_number)
        _number_of_sp = random.randint(self.nsp_min, self.nsp_max)
        _spxs = sorted(random.sample(self.numbers, k=_number_of_sp))
        _spys = random.sample(self.numbers, k=_number_of_sp)
        _intervals = [[_spys[i],_spys[i+1]] for i in range(len(_spys)-1)]
        _spss = [random.choice([1,-1])] + [1 if _y[0] <= _y[1] else -1 for _y in _intervals] + [random.choice([1,-1])]
        quiz.quiz_identifier = hash(str(_spxs) + str(_spys) + str(_spss))
        # 正答の選択肢の生成
        _saddles = []
        for i in range(len(_spys)):
            if _spss[i]*_spss[i+1] < 0:
                if _spss[i] < 0:
                    _saddles.append([_spxs[i],_spys[i],-1])
                else:
                    _saddles.append([_spxs[i],_spys[i],1])
        quiz.data = [_number_of_sp, _spxs, _spys, _spss, _saddles]
        ans = { 'fraction': 100, 'data': _saddles }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_number_of_sp, _spxs, _spys, _spss, _saddles] = quiz.data
        if len(_saddles) > 0:
            ans['feedback'] = r'増減表をきちんと確認してください。極値をもちます。'
            ans['data'] = list()
            answers.append(dict(ans))
            ans['feedback'] = r'極大と極小の定義を確認しましょう。取り違えています。'
            ans['data'] = [[_saddle[0], _saddle[1], -1*_saddle[2]] for _saddle in _saddles]
            answers.append(dict(ans))
        if len(_saddles) > 1:
            ans['feedback'] = r'全ての極値を調べてください。不足があります。'
            for i in range(len(_saddles)):
                ans['data'] = _saddles[:i] + _saddles[i+1:]
                answers.append(dict(ans))
                ans['feedback'] = r'極大と極小の定義を確認しましょう。不足もありますし，取り違えてもいます。'
                ans['data'] = [[_saddle[0], _saddle[1], -1*_saddle[2]] for _saddle in ans['data']]
                answers.append(dict(ans))
        if len(_saddles) < len(_spxs):
            _used_spxs = [_saddle[0] for _saddle in _saddles]
            for i in range(_number_of_sp):
                if _spxs[i] in _used_spxs:
                    continue
                _incorrect_small = []
                _incorrect_large = []
                for _saddle in _saddles:
                    if _saddle[0] < _spxs[i]:
                        _incorrect_small.append(_saddle)
                    else:
                        _incorrect_large.append(_saddle)
                ans['feedback'] = r'極大と極小の定義を確認しましょう。多すぎます。'
                ans['data'] = _incorrect_small + [[_spxs[i], _spys[i], -1]] + _incorrect_large
                answers.append(dict(ans))
                ans['feedback'] = r'極大と極小の定義を確認しましょう。多すぎます。'
                ans['data'] = _incorrect_small + [[_spxs[i], _spys[i], +1]] + _incorrect_large
                answers.append(dict(ans))
            
#        answers = common.answer_union(answers)
        if len(answers) > size:
            answers = random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_number_of_sp, _spxs, _spys, _spss, _saddles] = quiz.data
        _text = r'以下は，ある関数\( f(x) \)の増減表です。\( f(x) \)に関して最も適切なものを選択してください。'
        _text += r'<br />\( \begin{array}{'
        for i in range(2*_number_of_sp+2):
            _text += r'|c'
        _text += r'|}\hline '
        _text += r'x'
        for _spx in _spxs:
            _text += r'&\cdots&' + sympy.latex(_spx)
        _text += r"&\cdots\\\hline f'(x)"
        for _sps in _spss[:-1]:
            if _sps > 0:
                _text += r'&+&0'
            else:
                _text += r'&-&0'
        if _spss[-1] > 0:
            _text += r'&+'
        else:
            _text += r'&-'
        _text += r'\\\hline f(x)'
        for _spy in _spys:
            _text += r'&&' + sympy.latex(_spy)
        _text += r"&\\\hline"
        _text += r'\end{array} \)'
        return _text
    def answer_text(self, ans):
        _saddles = ans['data']
        if len(_saddles) == 0:
            return r'極値をもたない'
        _text = r''
        for _saddle in _saddles[:-1]:
            _text += r'\( x=' + sympy.latex(_saddle[0]) + r' \)のとき'
            if _saddle[2] > 0:
                _text += r'極大値'
            else:
                _text += r'極小値'
            _text += r'\( ' +  sympy.latex(_saddle[1]) + r' \)，'
        _text += r'\( x=' + sympy.latex(_saddles[-1][0]) + r' \)のとき'
        if _saddles[-1][2] > 0:
            _text += r'極大値'
        else:
            _text += r'極小値'
        _text += r'\( ' +  sympy.latex(_saddles[-1][1]) + r' \)をとる'
        return _text


# In[125]:


if __name__ == "__main__":
    q = local_minimum_maximum()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=25)


# In[126]:


if __name__ == "__main__":
    pass
    #qz.save('local_minimum_maximum.xml')


# ## inflection points

# In[247]:


class inflection_points(core.Question):
    name = '変曲点（増減表からの決定）'    
    def __init__(self, emax=5, nmin=1, nmax=2, srate=0.5):
        # 生成するdf=0の個数範囲
        self.nsp_min = nmin
        self.nsp_max = nmax
        # 左端と右端に変曲点がある確率
        self.slope_ratio = srate
        # 数の生成
        self.numbers = list(range(0,emax+1)) + [sympy.sqrt(sympy.prime(i)) for i in range(1,emax)] + [sympy.pi, sympy.E]
        self.numbers = self.numbers + [_e/sympy.Integer(2) for _e in self.numbers]
        self.numbers = self.numbers + [-_e for _e in self.numbers]
        self.numbers = sorted(list(set(self.numbers)))
    def random_interval(self, x0, x1):
        if x0 == sympy.nan:
            _i0 = -1
        else:
            _i0 = self.numbers.index(x0)
        if x1 == sympy.nan:
            _i1 = len(self.numbers)
        else:
            _i1 = self.numbers.index(x1)
        if _i1 == 0:
            return self.numbers[0] - 1
        elif _i0 == len(self.numbers) - 1:
            return self.numbers[-1] + 1
        elif _i0 + 1 < _i1:
            return random.choice(self.numbers[_i0+1:_i1])
        else:
            return (x0+x1)/sympy.Integer(2)
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='変曲点', quiz_number=_quiz_number)
        _number_of_sp = random.randint(self.nsp_min, self.nsp_max)
        _spxs = sorted(random.sample(self.numbers, k=_number_of_sp))
        _spys = random.sample(self.numbers, k=_number_of_sp)
        _intervals = [[_spys[i],_spys[i+1]] for i in range(len(_spys)-1)]
        _spss = [random.choice([1,-1])] + [1 if _y[0] <= _y[1] else -1 for _y in _intervals] + [random.choice([1,-1])]
        _matrix = sympy.zeros(4,4*_number_of_sp-1)*sympy.nan
        for i in range(_number_of_sp):
            _matrix[0,4*i+1] = _spxs[i] 
            _matrix[1,4*i] = _spss[i]
            _matrix[1,4*i+1] = 0
            _matrix[1,4*i+2] = _spss[i+1]
            if _spss[i] == _spss[i+1]:
                _matrix[2,4*i+1] = 0
            _matrix[3,4*i+1] = _spys[i]
        for i in range(_number_of_sp-1):
            _matrix[0,4*i+3] = self.random_interval(_spxs[i],_spxs[i+1])
            _matrix[1,4*i+3] = _spss[i+1]
            _matrix[2,4*i+3] = 0
            _matrix[3,4*i+3] = self.random_interval(_spys[i],_spys[i+1])
        if random.random() < self.slope_ratio:
            _matrix = _matrix.col_insert(0, sympy.Matrix([self.random_interval(sympy.nan,_spxs[0]), _spss[0], 0, self.random_interval(sympy.nan,_spys[0])]))
            _matrix = _matrix.col_insert(0, sympy.Matrix([sympy.nan, _spss[0], sympy.nan, sympy.nan]))
        if random.random() < self.slope_ratio:
            _matrix = _matrix.col_insert(_matrix.cols, sympy.Matrix([self.random_interval(_spxs[-1], sympy.nan), _spss[-1], 0, self.random_interval(_spys[-1], sympy.nan)]))
            _matrix = _matrix.col_insert(_matrix.cols, sympy.Matrix([sympy.nan, _spss[-1], sympy.nan, sympy.nan]))
        _preserved = []
        for i in range(_matrix.cols):
            if _matrix[2,i] != 0:
                _preserved.append(i)
            if _matrix[2,i] == 0 or i == _matrix.cols - 1:
                if len(_preserved) > 1:
                    if _matrix[1,_preserved[0]] > 0 and _matrix[1,_preserved[-1]] < 0:
                        for j in _preserved:
                            _matrix[2,j] = -1
                        _preserved = []
                        continue
                    elif _matrix[1,_preserved[0]] < 0 and _matrix[1,_preserved[-1]] > 0:
                        for j in _preserved:
                            _matrix[2,j] = 1
                        _preserved = []
                        continue
                for j in _preserved:
                    is_left_zero = False
                    is_right_zero = False
                    if j != 0:
                        if _matrix[1,j-1] == 0:
                            is_left_zero = True
                    if j != _matrix.cols - 1:
                        if _matrix[1,j+1] == 0:
                            is_right_zero = True
                    if is_left_zero:
                        _matrix[2,j] = _matrix[1,j]
                    if is_right_zero:
                        _matrix[2,j] = -1*_matrix[1,j]       
                _preserved = []
        if not _matrix[2,0].is_real:
            _matrix[2,0] = _matrix[1,0]
        if not _matrix[2,_matrix.cols - 1].is_real:
            _matrix[2,_matrix.cols - 1] = -1*_matrix[1,_matrix.cols - 1]
        quiz.quiz_identifier = hash(str(_matrix))
        # 正答の選択肢の生成
        _inflections = []
        for i in range(_matrix.cols):
            if _matrix[2,i] == 0:
                if _matrix[2,i-1]*_matrix[2,i+1] < 0:
                    _inflections.append(_matrix[0,i])
        quiz.data = [_matrix, _inflections]
        ans = { 'fraction': 100, 'data': _inflections }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_matrix, _inflections] = quiz.data
        if len(_inflections) > 0:
            ans['feedback'] = r'増減表をきちんと確認してください。変曲点をもちます。'
            ans['data'] = list()
            answers.append(dict(ans))
        _incorrect = []
        for i in range(_matrix.cols):
            if _matrix[1,i] == 0:
                if _matrix[1,i-1]*_matrix[1,i+1] < 0:
                    _incorrect.append(_matrix[0,i])
        if _incorrect != _inflections:
            ans['feedback'] = r'変曲点と極値をもつ点を混同しないようにしましょう。'
            ans['data'] = _incorrect
            answers.append(dict(ans))
        _incorrect = []
        for i in range(_matrix.cols):
            if _matrix[1,i] == 0:
                _incorrect.append(_matrix[0,i])
        if _incorrect != _inflections:
            ans['feedback'] = r'変曲点と極値の必要条件の点を混同しないようにしましょう。'
            ans['data'] = _incorrect
            answers.append(dict(ans))
        _incorrect = []
        for i in range(_matrix.cols):
            if _matrix[2,i] == 0:
                _incorrect.append(_matrix[0,i])
        if _incorrect != _inflections:
            ans['feedback'] = r'変曲点の定義を確認しましょう。'
            ans['data'] = _incorrect
            answers.append(dict(ans))
        answers = common.answer_union(answers)
        additional_answers = []
        ans['feedback'] = r'変曲点の定義や増減表の見方などを確認してください。'
        _incorrect_numbers = []
        for i in range(_matrix.cols):
            if _matrix[0,i].is_real:
                _incorrect_numbers.append(_matrix[0,i])
                if _inflections != [_matrix[0,i]]:
                    ans['data'] = [_matrix[0,i]]
                    additional_answers.append(dict(ans))
            if _matrix[3,i].is_real:
                _incorrect_numbers.append(_matrix[3,i])
                if _inflections != [_matrix[3,i]]:
                    ans['data'] = [_matrix[3,i]]
                    additional_answers.append(dict(ans))                    
        if len(_incorrect_numbers) < 2:
            _incorrect_numbers.append(_incorrect_numbers[0]+1)
        for i in range(size):
            _incorrect = sorted(random.sample(_incorrect_numbers, k=2))
            if _incorrect != _inflections:
                ans['data'] = _incorrect
                additional_answers.append(dict(ans))                    
        additional_answers = common.answer_union(additional_answers)
        if len(additional_answers) > len(answers):
            additional_answers = random.sample(additional_answers, k=max(len(answers),min(size-len(answers),len(additional_answers))))
        answers = answers + additional_answers
        answers = common.answer_union(answers)
        if len(answers) > size:
            answers = random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_matrix, _inflections] = quiz.data
        _text = r'以下は，ある関数\( f(x) \)の増減表です。\( f(x) \)に関して最も適切なものを選択してください。'
        _text += r'<br />\( \begin{array}{'
        for i in range(_matrix.cols + 1):
            _text += r'|c'
        _text += r'|}\hline '
        _text += r'x'
        for i in range(_matrix.cols):
            if _matrix[0,i].is_real:
                _text += r'&' + sympy.latex(_matrix[0,i])
            else:
                _text += r'&\cdots'
        _text += r"\\\hline f'(x)"
        for i in range(_matrix.cols):
            if _matrix[1,i].is_real:
                if _matrix[1,i] > 0:
                    _text += r'&+'
                elif _matrix[1,i] < 0:
                    _text += r'&-'
                else:
                    _text += r'&0'
            else:
                _text += r'&'
        _text += r"\\\hline f''(x)"
        for i in range(_matrix.cols):
            if _matrix[2,i].is_real:
                if _matrix[2,i] > 0:
                    _text += r'&+'
                elif _matrix[2,i] < 0:
                    _text += r'&-'
                else:
                    _text += r'&0'
            else:
                _text += r'&'
        _text += r'\\\hline f(x)'
        for i in range(_matrix.cols):
            if _matrix[3,i].is_real:
                _text += r'&' + sympy.latex(_matrix[3,i])
            else:
                _text += r'&'
        _text += r"\\\hline"
        _text += r'\end{array} \)'
        return _text
    def answer_text(self, ans):
        _inflections = ans['data']
        if len(_inflections) == 0:
            return r'変曲点をもたない'
        _text = r'\( x='
        for _x in _inflections:
            _text += sympy.latex(_x) + r','
        _text = _text[:-1]
        _text += r' \)で変曲点をもつ'
        return _text


# In[248]:


if __name__ == "__main__":
    q = inflection_points()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=5)


# In[249]:


if __name__ == "__main__":
    pass
    #qz.save('inflection_points.xml')


# ## power series expansion

# In[5]:


class power_series_expansion(core.Question):
    name = '級数展開（マクローリンとテイラー展開）'
    _x = sympy.Symbol('x')
    def __init__(self, emin=-3, emax=3, nmin=2, nmax=4, amin=-2, amax=2, azero=True, omin=1, omax=3):
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 展開点の範囲
        self.epnt_min = amin
        self.epnt_max = amax
        self.epnt_zero = azero
        # 多項式関数の範囲
        self.deg_min = nmin
        self.deg_max = nmax
        # 展開次数
        self.ord_min = omin
        self.ord_max = omax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='級数展開', quiz_number=_quiz_number)
        _type = random.choice(['polynomial', 'sin', 'cos', 'exp', 'aexp', 'rational', 'log'])
        while _type == 'log' and self.epnt_min == self.epnt_max and self.epnt_min == 0:
            _type = random.choice(['polynomial', 'sin', 'cos', 'exp', 'aexp', 'rational', 'log'])
        if _type == 'polynomial':
            _func = sympy.Integer(0)
            while _func == 0 or _func.is_real:
                for i in range(random.randint(self.deg_min, self.deg_max)+1):
                    _func = _func + random.randint(self.elem_min, self.elem_max)*self._x**i
            if self.epnt_zero:
                _point = random.randint(self.epnt_min, self.epnt_max)
            else:
                _point = nonzero_randint(self.epnt_min, self.epnt_max)
        elif _type == 'sin':
            _func = nonzero_randint(self.elem_min, self.elem_max)*sympy.sin(nonzero_randint(self.elem_min, self.elem_max)*self._x)
            _point = sympy.pi*random.randint(3*self.epnt_min, 3*self.epnt_max)/sympy.Integer(3)
            while _point == 0 and not self.epnt_zero:
                _point = sympy.pi*random.randint(3*self.epnt_min, 3*self.epnt_max)/sympy.Integer(3)
        elif _type == 'cos':
            _func = nonzero_randint(self.elem_min, self.elem_max)*sympy.sin(nonzero_randint(self.elem_min, self.elem_max)*self._x)
            _point = sympy.pi*random.randint(3*self.epnt_min, 3*self.epnt_max)/sympy.Integer(3)
            while _point == 0 and not self.epnt_zero:
                _point = sympy.pi*random.randint(3*self.epnt_min, 3*self.epnt_max)/sympy.Integer(3)
        elif _type == 'exp':
            _func = nonzero_randint(self.elem_min, self.elem_max)*sympy.exp(nonzero_randint(self.elem_min, self.elem_max)*self._x)
            _point = random.randint(self.epnt_min, self.epnt_max)
            while _point == 0 and not self.epnt_zero:
                _point = random.randint(self.epnt_min, self.epnt_max)
        elif _type == 'aexp':
            _func = nonzero_randint(self.elem_min, self.elem_max)*nonone_randpint(self.elem_min, self.elem_max)**(nonzero_randint(self.elem_min, self.elem_max)*self._x)
            _point = random.randint(self.epnt_min, self.epnt_max)
            while _point == 0 and not self.epnt_zero:
                _point = random.randint(self.epnt_min, self.epnt_max)
        elif _type == 'log':
            _func = nonzero_randint(self.elem_min, self.elem_max)*sympy.log(abs(nonzero_randint(self.elem_min, self.elem_max))*self._x)
            _point = abs(nonzero_randint(self.epnt_min, self.epnt_max))
            while _point == 0 and not self.epnt_zero:
                _point = abs(nonzero_randint(self.epnt_min, self.epnt_max))
        else:
            _point = random.randint(self.epnt_min, self.epnt_max)
            while _point == 0 and not self.epnt_zero:
                _point = random.randint(self.epnt_min, self.epnt_max)
            _func = sympy.Integer(0)
            while _func.subs(self._x, _point) == 0:
                _func = sympy.Integer(0)
                while _func == 0 or _func.is_real:
                    for i in range(2):
                        _func = _func + random.randint(self.elem_min, self.elem_max)*self._x**i
            _func = sympy.Integer(1)/_func
        _order = random.randint(self.ord_min, self.ord_max)
        quiz.quiz_identifier = hash(str(_func) + str(_point) + str(_order))
        # 正答の選択肢の生成
        _series = _func.series(x=self._x, x0=_point, n=_order+1).removeO()        
        quiz.data = [_func, _point, _order, _series]
        ans = { 'fraction': 100, 'data': _series }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def _incorrect_series_expansion_by_factorial(self, _func, _point, _order):
        _incorrect = 0
        for i in range(_order+1):
            _incorrect = _incorrect + sympy.diff(_func,self._x,i).subs(self._x,_point)*(self._x-_point)**i
        return _incorrect
    def _incorrect_series_expansion_by_factorial2(self, _func, _point, _order):
        _incorrect = 0
        for i in range(_order+1):
            _incorrect = _incorrect + (1 if i == 0 else 1/sympy.Integer(i))*sympy.diff(_func,self._x,i).subs(self._x,_point)*(self._x-_point)**i
        return _incorrect
    def _incorrect_series_expansion_by_signofa(self, _func, _point, _order):
        _incorrect = 0
        for i in range(_order+1):
            _incorrect = _incorrect + (sympy.factorial(i))*sympy.diff(_func,self._x,i).subs(self._x,_point)*(self._x+_point)**i
        return _incorrect
    def _incorrect_series_expansion_by_factorial_and_signofa(self, _func, _point, _order):
        _incorrect = 0
        for i in range(_order+1):
            _incorrect = _incorrect + sympy.diff(_func,self._x,i).subs(self._x,_point)*(self._x+_point)**i
        return _incorrect
    def _incorrect_series_expansion_by_factorial2_and_signofa(self, _func, _point, _order):
        _incorrect = 0
        for i in range(_order+1):
            _incorrect = _incorrect + (1 if i == 0 else 1/sympy.Integer(i))*sympy.diff(_func,self._x,i).subs(self._x,_point)*(self._x+_point)**i
        return _incorrect
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_func, _point, _order, _series] = quiz.data
        for _n in range(2,_order+2):
            _incorrect = _func.series(x=self._x, x0=_point, n=_n).removeO()   
            if _incorrect != _series:
                ans['feedback'] = r'剰余項を除き，指定された次数の多項式として近似してください。'
                ans['data'] = _incorrect
                answers.append(dict(ans))
            _incorrect = self._incorrect_series_expansion_by_factorial(_func, _point, _n)   
            if _incorrect != _series:
                ans['feedback'] = r'展開に関する公式を確認しましょう。特に階乗部分を確認してください。'
                ans['data'] = _incorrect
                answers.append(dict(ans))
            _incorrect = self._incorrect_series_expansion_by_factorial2(_func, _point, _n)   
            if _incorrect != _series:
                ans['feedback'] = r'展開に関する公式を確認しましょう。特に階乗部分を確認してください。'
                ans['data'] = _incorrect
                answers.append(dict(ans))
        if _point != 0:
            for _n in range(2,_order+2):
                _incorrect = self._incorrect_series_expansion_by_signofa(_func, _point, _n)   
                if _incorrect != _series:
                    ans['feedback'] = r'展開に関する公式を確認しましょう。特に展開点に関する確認してください。'
                    ans['data'] = _incorrect
                    answers.append(dict(ans))
                _incorrect = self._incorrect_series_expansion_by_factorial_and_signofa(_func, _point, _n)   
                if _incorrect != _series:
                    ans['feedback'] = r'展開に関する公式を確認しましょう。かなり誤解している可能性があります。'
                    ans['data'] = _incorrect
                    answers.append(dict(ans))
                _incorrect = self._incorrect_series_expansion_by_factorial2_and_signofa(_func, _point, _n)   
                if _incorrect != _series:
                    ans['feedback'] = r'展開に関する公式を確認しましょう。かなり誤解している可能性があります。'
                    ans['data'] = _incorrect
                    answers.append(dict(ans))
        answers = common.answer_union(answers)
        while len(answers) < size:
            _incorrect = _series + random.randint(self.elem_min, self.elem_max)
            if _incorrect != _series:
                ans['feedback'] = r'展開に関する公式を確認しましょう。'
                ans['data'] = _incorrect
                answers.append(dict(ans))
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_func, _point, _order, _series] = quiz.data
        _text = r'次の関数\( f(x) \)の'
        if _point == 0:
            _text += r'マクローリン展開による'
        else:
            _text += r'\( x=' + sympy.latex(_point) + r' \)のまわりでのテイラー展開による'
        _text += r'\( ' + sympy.latex(_order) + r' \)次多項式としての近似を選択してください。<br />'
        _text += r'\( f(x)=' + sympy.latex(_func,  order='lex') + r' \)'
        return _text
    def answer_text(self, ans):
        return r'\( ' +  sympy.latex(ans['data'], order='lex', long_frac_ratio=1) + r' \)'


# In[6]:


if __name__ == "__main__":
    q = power_series_expansion(emin=-4, emax=4, amin=0, amax=0)
    qz = core.generate(q, size=200, category=r'マクローリン展開')
    qz.preview(size=25)


# In[7]:


if __name__ == "__main__":
    pass
    #qz.save('power_series_expansion_maclaurin.xml')


# In[8]:


if __name__ == "__main__":
    q = power_series_expansion(emin=-3, emax=3, amin=-2, amax=2, azero=False)
    qz = core.generate(q, size=200, category=r'テイラー展開')
    qz.preview(size=25)


# In[9]:


if __name__ == "__main__":
    pass
    #qz.save('power_series_expansion_taylor.xml')


# ## dummy

# In[ ]:





# # All the questions

# In[ ]:


questions_str = ['composite_function', 'limit_of_polynomial_or_rational_function', 
                 'limit_of_polynomial_or_rational_function_with_nolimit', 'directional_limit_of_rational_function', 
                 'asymptotic_behavior_of_limit', 'limit_of_trigonometric_function', 'limit_of_exponential_and_logarithmic_function',
                 'average_rate_of_change', 'differential_coefficient', 'differential_coefficient_intermediate_expression',
                 'derivative_by_definition', 'derivative_by_linearity', 'derivative_by_multiplication', 'derivative_by_quotient', 
                 'derivative_by_composition', 'derivative_by_taking_logarithm', 'higher_derivative', 'local_minimum_maximum', 
                 'inflection_points', 'power_series_expansion']
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




