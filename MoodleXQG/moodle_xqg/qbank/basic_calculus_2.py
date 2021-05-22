#!/usr/bin/env python
# coding: utf-8

# Moodle XML Question Generator, (C) 2020 Kosaku Nagasaka (Kobe University)

# ## モジュールファイルの生成

# In[1]:


if __name__ == "__main__":
    import subprocess
    subprocess.run(['jupyter', 'nbconvert', '--to', 'python', '_basic_calculus_2.ipynb','--output','basic_calculus_2.py'])


# # Basic Calculus 2 (differencial calculus for univariate function)

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
    core._force_to_this_lang = 'ja'
else:
    from .. import core
    from . import common


# In[1]:


import sympy
import random
import IPython
import itertools
import copy
import timeout_decorator


# ## minor helpers

# In[3]:


def nonzero_randint(imin, imax):
    if 0 < imin or imax < 0:
        return random.randint(imin, imax)
    else:
        return random.choice(list(set(range(imin,imax+1)).difference(set([0]))))


# In[4]:


def nonone_randpint(imin, imax):
    if 1 < imin:
        return random.randint(imin, imax)
    else:
        return abs(random.choice(list(set(range(imin,imax+1)).difference(set([0,-1,1])))))


# ## IntegrableFunction class for the following quizzes

# In[131]:


class IntegrableFunction():
    _const = sympy.Symbol('c') # non-zero integer
    _base = sympy.Symbol('a') # non-one positive integer
    _n = sympy.Symbol('n') # positive integer
    _x = sympy.Symbol('x')
    # in sympy, there is no way to represent log_a(x) without expanding.....
    # format: [f, F, incorrect Fs]
    _function_types = ['constant', 'monomial', 'rational', 'sine', 'cosine', 'tangent', 
                       'natural_exponent', 'general_exponent', 'natural_logarithm']
    _function_defs = dict()
    _function_defs['constant'] = [_const, _const*_x, [_const, 0, 1]]
    _function_defs['monomial'] = [_x**_n, 1/(_n+1)*_x**(_n+1), [_x**_n, (_n)*_x**(_n-1)]]
    _function_defs['rational'] = [_x**(-_base), 1/(-_base+1)*_x**(-_base+1), [_x**(-_base), (-_base)*_x**(-_base-1)]]
    _function_defs['sine'] = [sympy.sin(_x), -sympy.cos(_x), [sympy.sin(_x), sympy.cos(_x)]]
    _function_defs['cosine'] = [sympy.cos(_x), sympy.sin(_x), [sympy.cos(_x), -sympy.sin(_x)]]
    _function_defs['tangent'] = [1/sympy.cos(_x)**2, sympy.tan(_x), [1/sympy.cos(_x), 1/sympy.sin(_x)**2]]
    _function_defs['natural_exponent'] = [sympy.exp(_x), sympy.exp(_x), [sympy.exp(_x-1), sympy.exp(_x+1)]]
    _function_defs['general_exponent'] = [_base**_x, 1/(sympy.log(_base))*_base**_x, [_base**_x, 1/(sympy.log(_base))*_base**(_x+1)]]
    _function_defs['natural_logarithm'] = [1/_x, sympy.log(abs(_x)), [1/_x**2, -1/_x**2]]
    # func = [scalar*linearity]+
    # linearity = elemental_func | elemental_func + constant
    _incorrect_reasons = ['formula', 'sign', 'scalar', 'part']
    def __init__(self, emin=-3, emax=3, nmin=1, nmax=3, lmin=1, lmax=2, srate=0.25):
        # 生成する個々の係数等の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 生成する次数の範囲
        self.deg_min = nmin
        self.deg_max = nmax
        # 線形性に表れる関数の個数
        self.linearity_min = lmin
        self.linearity_max = lmax
        # 定数項の確率
        self.constant_term_ratio = srate
        # internals
        self.function = 0
        self.is_trigonometric = False
        self.is_interval_should_be_positive = False
        self.is_interval_should_be_less_pi_2 = False
        self.interval = []
    # func = [scalar*linearity]+
    def generate_function(self):
        while self.function == 0:
            _n = random.randint(self.linearity_min, self.linearity_max)
            _func1 = self._generate_linearity()
            if _n == 1:
                self.function = _func1
                return
            _func = ['summation', _func1]
            for _i in range(_n-1):
                _func.append(self._generate_linearity())
            self.function = _func
    # linearity = elemental_func | elemental_func + constant
    def _generate_linearity(self, _type=None):
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
        if _type in ['sine', 'cosine', 'tangent']:
            self.is_trigonometric = True
        if _type in ['tangent']:
            self.is_interval_should_be_less_pi_2 = True
        if _type in ['monomial', 'rational', 'natural_logarithm']:
            self.is_interval_should_be_positive = True            
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
        # ['summation', 'scalar']  
        if _func[0] == 'summation':
            _summand = 0
            for _f in _func[1:]:
                _summand = _summand + _recursive_call(_f)
            return _summand
        elif _func[0] == 'scalar':
            return _func[1] * _recursive_call(_func[2])
        else:
            return _func[0]
    def get_function(self):
        if self.function == 0:
            self.generate_function()
        return self._get_function(self.function)
    def generate_interval(self):
        if self.is_trigonometric:
            _candidates = [-sympy.pi, -sympy.pi/2, -sympy.pi/3, -sympy.pi/4, 0, sympy.pi/4, sympy.pi/3, sympy.pi/2, sympy.pi]
        else:
            _candidates = [sympy.Integer(_elem) for _elem in range(self.elem_min,self.elem_max+1)]
        if self.is_interval_should_be_less_pi_2:
            _candidates = [_elem for _elem in _candidates if abs(_elem) < sympy.pi/2]
        if self.is_interval_should_be_positive:
            _candidates = [_elem for _elem in _candidates if _elem > 0]
        self.interval = sorted(random.sample(_candidates,2))
    def _get_definite_integral(self, _func, _interval=None):
        if _interval is None:
            _interval = self.interval
        if isinstance(_func, list):
            _func = self._get_function(_func)
        return sympy.integrate(_func, (self._x, _interval[0], _interval[1]))
        if isinstance(_func, list):
            _indefinite_integral = self._get_antiderivative(_func)
        else:
            _indefinite_integral = sympy.integrate(_func, self._x)
        _fa = self._mysubs(_indefinite_integral, [self._x], [_interval[0]])        
        _fb = self._mysubs(_indefinite_integral, [self._x], [_interval[1]])
        return _fb - _fa
    def get_interval(self):
        if self.function == 0:
            self.generate_function()
        if len(self.interval) == 0:
            self.generate_interval()
        return self.interval        
    def get_definite_integral(self):
        if self.function == 0:
            self.generate_function()
        if len(self.interval) == 0:
            self.generate_interval()
        return self._get_definite_integral(self.function)
    def _get_antiderivative(self, _func):
        _recursive_call = self._get_antiderivative
        # ['summation', 'scalar']  
        if _func[0] == 'summation':
            _summand = 0
            for _f in _func[1:]:
                _summand = _summand + _recursive_call(_f)
            return _summand
        elif _func[0] == 'scalar':
            return _func[1] * _recursive_call(_func[2])
        else:
            return _func[1]
    def get_antiderivative(self):
        if self.function == 0:
            self.generate_function()
        return self._get_antiderivative(self.function)
    def _get_incorrect_antiderivatives_by_formula(self, _func):
        _recursive_call = self._get_incorrect_antiderivatives_by_formula
        if _func[0] == 'summation':
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
        else:
            return _func[2]
    def _get_incorrect_antiderivatives_by_sign(self, _func):
        _recursive_call = self._get_incorrect_antiderivatives_by_sign
        if _func[0] == 'summation':
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
        else:
            return [-_func[1]]
    def _get_incorrect_antiderivatives_by_scalar(self, _func):
        _recursive_call = self._get_incorrect_antiderivatives_by_scalar
        if _func[0] == 'summation':
            _summand = [0]
            for _f in _func[1:]:
                _dfs = _recursive_call(_f)
                _new_summand = []
                for _df in _dfs:
                    _new_summand = _new_summand + [ _s + _df for _s in _summand]
                _summand = _new_summand
            return _summand
        elif _func[0] == 'scalar':
            return [_f for _f in _recursive_call(_func[2])]
        else:
            return [_func[1]]
    def _get_incorrect_antiderivatives_by_part(self, _func):
        _recursive_call = self._get_incorrect_antiderivatives_by_part
        # ['summation', 'scalar']  
        if _func[0] == 'summation':
            _funcs = []
            for _f in _func[1:]:
                _funcs = _funcs + _recursive_call(_f)
            return _funcs
        elif _func[0] == 'scalar':
            return [_func[1] * _f for _f in _recursive_call(_func[2])]
        else:
            return [_func[1]]
    def _not_same_check(self, value):
        if not value.is_real:
            return False
        if value > 1.0e-4:
            return True
        return False
    def _incorrect_antiderivarives_only(self, _dfs):
        _funcs = []
        _correct_df = self.get_antiderivative()
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
    def get_incorrect_antiderivatives(self, _type=None):
        if self.function == 0:
            self.generate_function()
        if _type is None:
            _type = random.choice(self._incorrect_reasons)
        if _type == 'formula':
            _dfs = self._get_incorrect_antiderivatives_by_formula(self.function)
        elif _type == 'sign':
            _dfs = self._get_incorrect_antiderivatives_by_sign(self.function)
        elif _type == 'scalar':
            _dfs = self._get_incorrect_antiderivatives_by_scalar(self.function)
        elif _type == 'part':
            _dfs = self._get_incorrect_antiderivatives_by_part(self.function)
        return self._incorrect_antiderivarives_only(_dfs)
    def _incorrect_definite_integrals_only(self, _dfs):
        _dints = []
        _correct_dint = self.get_definite_integral()
        for _df in _dfs:
            _dint = self._get_definite_integral(_df)            
            if _dint in _dints:
                continue
            if self._not_same_check(abs(sympy.N(_correct_dint - _dint))):
                _dints.append(_dint)
        return _dints
    def get_incorrect_definite_integrals(self, _type=None):
        if self.function == 0:
            self.generate_function()
        if len(self.interval) == 0:
            self.generate_interval()
        if _type is None:
            _type = random.choice(self._incorrect_reasons)
        if _type == 'formula':
            _dfs = self._get_incorrect_antiderivatives_by_formula(self.function)
        elif _type == 'sign':
            _dfs = self._get_incorrect_antiderivatives_by_sign(self.function)
        elif _type == 'scalar':
            _dfs = self._get_incorrect_antiderivatives_by_scalar(self.function)
        elif _type == 'part':
            _dfs = self._get_incorrect_antiderivatives_by_part(self.function)
        return self._incorrect_definite_integrals_only(_dfs)


# In[132]:


if __name__ == "__main__":
    antidf = IntegrableFunction()
    antidf.generate_function()
    antidf.generate_interval()
    display(antidf.get_function())
    display(antidf.get_antiderivative())
    display(antidf.get_interval())
    display(antidf.get_definite_integral())
    display(antidf.get_incorrect_definite_integrals())


# ## antiderivative by linearity

# In[76]:


class antiderivative_by_linearity(core.Question):
    name = '線形性に基づく不定積分の計算（基本公式と線形性の活用）'
    def __init__(self, emin=-3, emax=3, nmin=1, nmax=3, lmin=2, lmax=3, srate=0.25):
        # 生成する個々の係数等の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 生成する次数の範囲
        self.deg_min = nmin
        self.deg_max = nmax
        # 生成する関数の和の範囲
        self.func_min = lmin
        self.func_max = lmax
        # コンスタントの割合
        self.constant_rate = srate
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='線形性に基づく不定積分の計算', quiz_number=_quiz_number)
        antidf = IntegrableFunction(emin=self.elem_min, emax=self.elem_max, nmin=self.deg_min, nmax=self.deg_max, 
                                   lmin=self.func_min, lmax=self.func_max, srate=self.constant_rate)        
        _func = antidf.get_function()
        quiz.quiz_identifier = hash(str(_func))
        # 正答の選択肢の生成
        _antidf = antidf.get_antiderivative()
        quiz.data = [_func, _antidf, antidf]
        ans = { 'fraction': 100, 'data': _antidf }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_func, _antidf, antidf] = quiz.data
        ans['feedback'] = r'基本的な関数の不定積分の公式を確認してください。'
        for _incorrect in antidf.get_incorrect_antiderivatives('formula'):
            ans['data'] = _incorrect
            answers.append(dict(ans))
        ans['feedback'] = r'不定積分の線形性に基づいて計算してください。定数倍や加減算を忘れている可能性があります。'
        for _incorrect in antidf.get_incorrect_antiderivatives('scalar'):
            ans['data'] = _incorrect
            answers.append(dict(ans))
        for _incorrect in antidf.get_incorrect_antiderivatives('part'):
            ans['data'] = _incorrect
            answers.append(dict(ans))
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_func, _antidf, antidf] = quiz.data
        _text = r'次の不定積分を求めた結果を選択してください。なお，\( C \)を積分定数とします。'
        _text += r'<br />\( \displaystyle\int \left(' + sympy.latex(_func, order='lex') + r'\right) dx \)'
        return _text
    def answer_text(self, ans):
        return r'\( ' +  sympy.latex(ans['data'], order='lex') + r' +C \)'


# In[126]:


if __name__ == "__main__":
    q = antiderivative_by_linearity(lmin=1, lmax=1, srate=1.0)
    q.name = '線形性に基づく不定積分の計算（基本公式の活用）'
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=25)


# In[78]:


if __name__ == "__main__":
    pass
    #qz.save('derivative_by_linearity_small.xml')


# In[79]:


if __name__ == "__main__":
    q = antiderivative_by_linearity()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=25)


# In[80]:


if __name__ == "__main__":
    pass
    #qz.save('derivative_by_linearity.xml')


# ## definite integral by linearity

# In[133]:


class definite_integral_by_linearity(core.Question):
    name = '線形性に基づく定積分の計算（基本公式と線形性の活用）'
    def __init__(self, emin=-3, emax=3, nmin=1, nmax=3, lmin=2, lmax=3, srate=0.25):
        # 生成する個々の係数等の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 生成する次数の範囲
        self.deg_min = nmin
        self.deg_max = nmax
        # 生成する関数の和の範囲
        self.func_min = lmin
        self.func_max = lmax
        # コンスタントの割合
        self.constant_rate = srate
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='線形性に基づく定積分の計算', quiz_number=_quiz_number)
        antidf = IntegrableFunction(emin=self.elem_min, emax=self.elem_max, nmin=self.deg_min, nmax=self.deg_max, 
                                   lmin=self.func_min, lmax=self.func_max, srate=self.constant_rate)        
        _func = antidf.get_function()
        quiz.quiz_identifier = hash(str(_func))
        # 正答の選択肢の生成
        _dint = antidf.get_definite_integral()
        _interval = antidf.get_interval()
        quiz.data = [_func, _dint, _interval, antidf]
        ans = { 'fraction': 100, 'data': _dint }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_func, _dint, _interval, antidf] = quiz.data
        if -_dint != _dint:
            ans['feedback'] = r'定積分の区間の端点における原始関数の値の引き算の向きを確認しましょう。'
            ans['data'] = -_dint
            answers.append(dict(ans))
        ans['feedback'] = r'基本的な関数の不定積分の公式を確認してください。それを用いて定積分を求めます。'
        for _incorrect in antidf.get_incorrect_definite_integrals('formula'):
            ans['data'] = _incorrect
            answers.append(dict(ans))
            if _incorrect != _dint:
                ans['data'] = -_incorrect
                answers.append(dict(ans))
        ans['feedback'] = r'不定積分の線形性に基づいて計算してください。定数倍や加減算を忘れている可能性があります。それを用いて定積分を求めます。'
        for _incorrect in antidf.get_incorrect_definite_integrals('scalar'):
            ans['data'] = _incorrect
            answers.append(dict(ans))
            if _incorrect != _dint:
                ans['data'] = -_incorrect
                answers.append(dict(ans))
        for _incorrect in antidf.get_incorrect_definite_integrals('part'):
            ans['data'] = _incorrect
            answers.append(dict(ans))
            if _incorrect != _dint:
                ans['data'] = -_incorrect
                answers.append(dict(ans))
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_func, _dint, _interval, antidf] = quiz.data
        _text = r'次の定積分を求めた結果を選択してください。'
        _text += r'<br />\( \displaystyle\int_{' + sympy.latex(_interval[0]) + r'}^{' + sympy.latex(_interval[1]) + r'} \left(' + sympy.latex(_func, order='lex') + r'\right) dx \)'
        return _text
    def answer_text(self, ans):
        return r'\( ' +  sympy.latex(ans['data'], order='lex') + r' \)'


# In[134]:


if __name__ == "__main__":
    q = definite_integral_by_linearity(lmin=1, lmax=1, srate=1.0)
    q.name = '線形性に基づく定積分の計算（基本公式の活用）'
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=25)


# In[135]:


if __name__ == "__main__":
    pass
    #qz.save('definite_integral_by_linearity_small.xml')


# In[138]:


if __name__ == "__main__":
    q = definite_integral_by_linearity(lmin=2, lmax=2, srate=1.0)
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=25)


# In[139]:


if __name__ == "__main__":
    pass
    #qz.save('definite_integral_by_linearity.xml')


# ## IntegrableBySubstitution class for the following quizzes

# In[9]:


class IntegrableBySubstitution():
    _const = sympy.Symbol('c') # non-zero integer
    _base = sympy.Symbol('a') # non-one positive integer
    _n = sympy.Symbol('n') # positive integer
    _x = sympy.Symbol('x', real=True)
    # in sympy, there is no way to represent log_a(x) without expanding.....
    # format: [f, F, incorrect Fs]
    _function_types = ['constant', 'monomial', 'rational', 'sine', 'cosine', 'tangent', 
                       'natural_exponent', 'general_exponent'] #, 'natural_logarithm']
    _function_defs = dict()
    _function_defs['constant'] = [_const, _const*_x, [_const, 0, 1]]
    _function_defs['monomial'] = [_x**_n, 1/(_n+1)*_x**(_n+1), [_x**_n, (_n)*_x**(_n-1)]]
    _function_defs['rational'] = [_x**(-_base), 1/(-_base+1)*_x**(-_base+1), [_x**(-_base), (-_base)*_x**(-_base-1)]]
    _function_defs['rational'] = _function_defs['monomial']
    _function_defs['sine'] = [sympy.sin(_x), -sympy.cos(_x), [sympy.sin(_x), sympy.cos(_x)]]
    _function_defs['cosine'] = [sympy.cos(_x), sympy.sin(_x), [sympy.cos(_x), -sympy.sin(_x)]]
    _function_defs['tangent'] = [1/sympy.cos(_x)**2, sympy.tan(_x), [1/sympy.cos(_x), 1/sympy.sin(_x)**2]]
    _function_defs['tangent'] = _function_defs['sine'] + _function_defs['cosine']
    _function_defs['natural_exponent'] = [sympy.exp(_x), sympy.exp(_x), [sympy.exp(_x-1), sympy.exp(_x+1)]]
    _function_defs['general_exponent'] = [_base**_x, 1/(sympy.log(_base))*_base**_x, [_base**_x, 1/(sympy.log(_base))*_base**(_x+1)]]
    _function_defs['general_exponent'] = _function_defs['natural_exponent']
    _function_defs['natural_logarithm'] = [1/_x, sympy.log(abs(_x)), [1/_x**2, -1/_x**2]]
    # func = [scalar*linearity]+
    # linearity = elemental_func | elemental_func + constant
    _incorrect_reasons = ['formula', 'sign', 'scalar', 'part']
    def __init__(self, emin=-3, emax=3, nmin=1, nmax=3, srate=0.25):
        # 生成する個々の係数等の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 生成する次数の範囲
        self.deg_min = nmin
        self.deg_max = nmax
        # 定数項の確率
        self.constant_term_ratio = srate
        # internals
        self.function = 0
        self.is_trigonometric = False
        self.is_interval_should_be_positive = False
        self.is_interval_should_be_less_pi_2 = False
        self.interval = []
        self.definite_integral = []
    # func = [scalar*linearity]+
    def generate_function(self):
        while self.function == 0:
            _type1 = random.choice(self._function_types[1:])
            _func1 = self._generate_linearity(_type1)
            if _type1 in ['monomial', 'rational']:
                _type2 = random.choice(self._function_types[3:])
            elif _type1 in ['natural_exponent', 'general_exponent']:
                _type2 = random.choice(self._function_types[3:6])
            else:
                _type2 = random.choice(self._function_types[1:])
            _func2 = self._generate_linearity(_type2)
            self.function = ['composition', _func1, _func2]
    # linearity = elemental_func | elemental_func + constant
    def _generate_linearity(self, _type=None):
        _scalar = nonzero_randint(self.elem_min, self.elem_max)
        _func1 = ['scalar', _scalar, self._generate_elemental_func(_type)]
        if random.random() >= self.constant_term_ratio:
            return _func1
        else:
            _func = ['summation', _func1]
            _func.append(self._generate_elemental_func('constant'))
        return _func
    def _mysubs(self, target, var, repl):
        if len(sympy.sympify(target).atoms(sympy.Symbol)) == 0:
            return target
        elif len(var) == 1:
            return target.subs(var[0], repl[0])
        else:
            return self._mysubs(target.subs(var[0], repl[0]), var[1:], repl[1:])
    def _generate_elemental_func(self, _type=None):
        if _type is None:
            _type = random.choice(self._function_types[1:])
        if _type in ['sine', 'cosine', 'tangent']:
            self.is_trigonometric = True
        if _type in ['tangent']:
            self.is_interval_should_be_less_pi_2 = True
        if _type in ['monomial', 'rational', 'natural_logarithm']:
            self.is_interval_should_be_positive = True            
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
        return sympy.diff(self._get_antiderivative(_func), self._x)
    def get_function(self):
        if self.function == 0:
            self.generate_function()
        return sympy.diff(self.get_antiderivative(), self._x)
    def generate_interval(self):
        if self.is_trigonometric:
            _candidates = [-sympy.pi, -sympy.pi/2, -sympy.pi/3, -sympy.pi/4, 0, sympy.pi/4, sympy.pi/3, sympy.pi/2, sympy.pi]
        else:
            _candidates = [sympy.Integer(_elem) for _elem in range(self.elem_min,self.elem_max+1)]
        if self.is_interval_should_be_less_pi_2:
            _candidates = [_elem for _elem in _candidates if abs(_elem) < sympy.pi/2]
        if self.is_interval_should_be_positive:
            _candidates = [_elem for _elem in _candidates if _elem > 0]
        self.interval = sorted(random.sample(_candidates,2))
    def _get_definite_integral(self, _func, _interval=None):
        if _interval is None:
            _interval = self.interval
        if isinstance(_func, list):
            _indefinite_integral = self._get_antiderivative(_func)
        else:
            _indefinite_integral = _func
        _fa = self._mysubs(_indefinite_integral, [self._x], [_interval[0]])        
        _fb = self._mysubs(_indefinite_integral, [self._x], [_interval[1]])
        return _fb - _fa
    def get_interval(self):
        if self.function == 0:
            self.generate_function()
        if len(self.interval) == 0:
            self.generate_interval()
        return self.interval        
    def get_definite_integral(self):
        if self.function == 0:
            self.generate_function()
        if len(self.interval) == 0:
            self.generate_interval()
        if isinstance(self.definite_integral, list):
            self.definite_integral = self._get_definite_integral(self.function)
        return self.definite_integral
    def _get_antiderivative(self, _func):
        _recursive_call = self._get_antiderivative
        # ['summation', 'scalar']  
        if _func[0] == 'summation':
            _summand = 0
            for _f in _func[1:]:
                _summand = _summand + _recursive_call(_f)
            return _summand
        elif _func[0] == 'composition':
            return _recursive_call(_func[1]).subs(self._x, _recursive_call(_func[2]))
        elif _func[0] == 'scalar':
            return _func[1] * _recursive_call(_func[2])
        else:
            return _func[1]
    def get_antiderivative(self):
        if self.function == 0:
            self.generate_function()
        return self._get_antiderivative(self.function)
    def _get_incorrect_antiderivatives_by_formula(self, _func):
        _recursive_call = self._get_incorrect_antiderivatives_by_formula
        if _func[0] == 'summation':
            _summand = [0]
            for _f in _func[1:]:
                _dfs = _recursive_call(_f)
                _new_summand = []
                for _df in _dfs:
                    _new_summand = _new_summand + [ _s + _df for _s in _summand]
                _summand = _new_summand
            return _summand
        elif _func[0] == 'composition':
            _funcs = []
            for _df1 in _recursive_call(_func[1]):
                for _df2 in _recursive_call(_func[2]):
                    _funcs.append(_df2 * self._mysubs(_df1, [self._x], [self._get_function(_func[2])]))
            return _funcs
        elif _func[0] == 'scalar':
            return [_func[1] * _f for _f in _recursive_call(_func[2])]
        else:
            return _func[2]
    def _get_incorrect_antiderivatives_by_sign(self, _func):
        _recursive_call = self._get_incorrect_antiderivatives_by_sign
        if _func[0] == 'summation':
            _summand = [0]
            for _f in _func[1:]:
                _dfs = _recursive_call(_f)
                _new_summand = []
                for _df in _dfs:
                    _new_summand = _new_summand + [ _s + _df for _s in _summand]
                _summand = _new_summand
            return _summand
        elif _func[0] == 'composition':
            _funcs = []
            for _df1 in _recursive_call(_func[1]):
                for _df2 in _recursive_call(_func[2]):
                    _funcs.append(_df2 * self._mysubs(_df1, [self._x], [self._get_function(_func[2])]))
            return _funcs
        elif _func[0] == 'scalar':
            return [_func[1] * _f for _f in _recursive_call(_func[2])]
        else:
            return [-_func[1]]
    def _get_incorrect_antiderivatives_by_scalar(self, _func):
        _recursive_call = self._get_incorrect_antiderivatives_by_scalar
        if _func[0] == 'summation':
            _summand = [0]
            for _f in _func[1:]:
                _dfs = _recursive_call(_f)
                _new_summand = []
                for _df in _dfs:
                    _new_summand = _new_summand + [ _s + _df for _s in _summand]
                _summand = _new_summand
            return _summand
        elif _func[0] == 'composition':
            _funcs = []
            for _df1 in _recursive_call(_func[1]):
                for _df2 in _recursive_call(_func[2]):
                    _funcs.append(_df2 * self._mysubs(_df1, [self._x], [self._get_function(_func[2])]))
            return _funcs
        elif _func[0] == 'scalar':
            return [_f for _f in _recursive_call(_func[2])]
        else:
            return [_func[1]]
    def _get_incorrect_antiderivatives_by_part(self, _func):
        _recursive_call = self._get_incorrect_antiderivatives_by_part
        # ['summation', 'scalar']  
        if _func[0] == 'summation':
            _funcs = []
            for _f in _func[1:]:
                _funcs = _funcs + _recursive_call(_f)
            return _funcs
        elif _func[0] == 'composition':
            _funcs = []
            for _df1 in _recursive_call(_func[1]):
                for _df2 in _recursive_call(_func[2]):
                    _funcs.append(_df2 * self._mysubs(_df1, [self._x], [self._get_function(_func[2])]))
            return _funcs
        elif _func[0] == 'scalar':
            return [_func[1] * _f for _f in _recursive_call(_func[2])]
        else:
            return [_func[1]]
    def _not_same_check(self, value):
        if not value.is_real:
            return False
        if value > 1.0e-4:
            return True
        return False
    def _incorrect_antiderivarives_only(self, _dfs):
        _funcs = []
        _correct_df = self.get_antiderivative()
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
    def get_incorrect_antiderivatives(self, _type=None):
        if self.function == 0:
            self.generate_function()
        if _type is None:
            _type = random.choice(self._incorrect_reasons)
        if _type == 'formula':
            _dfs = self._get_incorrect_antiderivatives_by_formula(self.function)
        elif _type == 'sign':
            _dfs = self._get_incorrect_antiderivatives_by_sign(self.function)
        elif _type == 'scalar':
            _dfs = self._get_incorrect_antiderivatives_by_scalar(self.function)
        elif _type == 'part':
            _dfs = self._get_incorrect_antiderivatives_by_part(self.function)
        return self._incorrect_antiderivarives_only(_dfs)
    @timeout_decorator.timeout(1)
    def _get_definite_integral_timeout(self, _df):
        return self._get_definite_integral(_df)
    def _incorrect_definite_integrals_only(self, _dfs):
        _dints = []
        _correct_dint = self.get_definite_integral()
        for _df in _dfs:
            try:
                _dint = self._get_definite_integral_timeout(_df)
            except:
                continue
            if _dint in _dints:
                continue
            if len(_dint.atoms(sympy.Symbol)) != 0:
                continue
            if self._not_same_check(abs(sympy.N(_correct_dint - _dint))):
                _dints.append(_dint)
        return _dints
    def get_incorrect_definite_integrals(self, _type=None):
        if self.function == 0:
            self.generate_function()
        if len(self.interval) == 0:
            self.generate_interval()
        if _type is None:
            _type = random.choice(self._incorrect_reasons)
        if _type == 'formula':
            _dfs = self._get_incorrect_antiderivatives_by_formula(self.function)
        elif _type == 'sign':
            _dfs = self._get_incorrect_antiderivatives_by_sign(self.function)
        elif _type == 'scalar':
            _dfs = self._get_incorrect_antiderivatives_by_scalar(self.function)
        elif _type == 'part':
            _dfs = self._get_incorrect_antiderivatives_by_part(self.function)
        return self._incorrect_definite_integrals_only(_dfs)


# In[22]:


if __name__ == "__main__":
    antidf = IntegrableBySubstitution()
    antidf.generate_function()
    antidf.generate_interval()
    display(antidf.get_function())
    display(antidf.get_antiderivative())
    display(antidf.get_incorrect_antiderivatives())
    display(antidf.get_interval())
    display(antidf.get_definite_integral())
    display(antidf.get_incorrect_definite_integrals())


# ## antiderivative by substitution

# In[36]:


class antiderivative_by_substitution(core.Question):
    name = '置換積分に基づく不定積分の計算（合成関数の微分法の逆）'
    def __init__(self, emin=-2, emax=2, nmin=1, nmax=2, srate=0.25):
        # 生成する個々の係数等の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 生成する次数の範囲
        self.deg_min = nmin
        self.deg_max = nmax
        # コンスタントの割合
        self.constant_rate = srate
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='置換積分に基づく不定積分の計算', quiz_number=_quiz_number)
        antidf = IntegrableBySubstitution(emin=self.elem_min, emax=self.elem_max, nmin=self.deg_min, nmax=self.deg_max, srate=self.constant_rate)        
        _func = antidf.get_function()
        quiz.quiz_identifier = hash(str(_func))
        # 正答の選択肢の生成
        _antidf = antidf.get_antiderivative()
        quiz.data = [_func, _antidf, antidf]
        ans = { 'fraction': 100, 'data': _antidf }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_func, _antidf, antidf] = quiz.data
        ans['feedback'] = r'基本的な関数の不定積分の公式を確認してください。'
        for _incorrect in antidf.get_incorrect_antiderivatives('formula'):
            ans['data'] = _incorrect
            answers.append(dict(ans))
        ans['feedback'] = r'不定積分の線形性に基づいて計算してください。定数倍や加減算を忘れている可能性があります。'
        for _incorrect in antidf.get_incorrect_antiderivatives('scalar'):
            ans['data'] = _incorrect
            answers.append(dict(ans))
        for _incorrect in antidf.get_incorrect_antiderivatives('part'):
            ans['data'] = _incorrect
            answers.append(dict(ans))
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_func, _antidf, antidf] = quiz.data
        _text = r'次の不定積分を求めた結果を選択してください。なお，\( C \)を積分定数とします。'
        _text += r'<br />\( \displaystyle\int \left(' + sympy.latex(_func, order='lex') + r'\right) dx \)'
        return _text
    def answer_text(self, ans):
        return r'\( ' +  sympy.latex(ans['data'], order='lex') + r' +C \)'


# In[ ]:


if __name__ == "__main__":
    q = antiderivative_by_substitution()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=200)


# In[38]:


if __name__ == "__main__":
    pass
    #qz.save('antiderivative_by_substitution.xml')


# ## definite integral by substitution

# In[26]:


class definite_integral_by_substitution(core.Question):
    name = '置換積分に基づく定積分の計算（合成関数の微分法の逆）'
    def __init__(self, emin=-2, emax=2, nmin=1, nmax=2, srate=0.25):
        # 生成する個々の係数等の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 生成する次数の範囲
        self.deg_min = nmin
        self.deg_max = nmax
        # コンスタントの割合
        self.constant_rate = srate
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='置換積分に基づく定積分の計算', quiz_number=_quiz_number)
        antidf = IntegrableBySubstitution(emin=self.elem_min, emax=self.elem_max, nmin=self.deg_min, nmax=self.deg_max, srate=self.constant_rate)        
        _func = antidf.get_function()
        quiz.quiz_identifier = hash(str(_func))
        # 正答の選択肢の生成
        _dint = antidf.get_definite_integral()
        _interval = antidf.get_interval()
        quiz.data = [_func, _dint, _interval, antidf]
        ans = { 'fraction': 100, 'data': _dint }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_func, _dint, _interval, antidf] = quiz.data
        if -_dint != _dint:
            ans['feedback'] = r'定積分の区間の端点における原始関数の値の引き算の向きを確認しましょう。'
            ans['data'] = -_dint
            answers.append(dict(ans))
        ans['feedback'] = r'基本的な関数の不定積分の公式を確認してください。それを用いて定積分を求めます。'
        for _incorrect in antidf.get_incorrect_definite_integrals('formula'):
            ans['data'] = _incorrect
            answers.append(dict(ans))
            if _incorrect != _dint:
                ans['data'] = -_incorrect
                answers.append(dict(ans))
        ans['feedback'] = r'不定積分の線形性に基づいて計算してください。定数倍や加減算を忘れている可能性があります。それを用いて定積分を求めます。'
        for _incorrect in antidf.get_incorrect_definite_integrals('scalar'):
            ans['data'] = _incorrect
            answers.append(dict(ans))
            if _incorrect != _dint:
                ans['data'] = -_incorrect
                answers.append(dict(ans))
        for _incorrect in antidf.get_incorrect_definite_integrals('part'):
            ans['data'] = _incorrect
            answers.append(dict(ans))
            if _incorrect != _dint:
                ans['data'] = -_incorrect
                answers.append(dict(ans))
        answers = common.answer_union(answers)
        while len(answers) < size:
            ans['feedback'] = r'原始関数を求め，公式に基づき計算を行ってください。'
            _incorrect = _dint + nonzero_randint(self.elem_min, self.elem_max)
            ans['data'] = _incorrect
            answers.append(dict(ans))            
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_func, _dint, _interval, antidf] = quiz.data
        _text = r'次の定積分を求めた結果を選択してください。'
        _text += r'<br />\( \displaystyle\int_{' + sympy.latex(_interval[0]) + r'}^{' + sympy.latex(_interval[1]) + r'} \left(' + sympy.latex(_func, order='lex') + r'\right) dx \)'
        return _text
    def answer_text(self, ans):
        return r'\( ' +  sympy.latex(ans['data'], order='lex') + r' \)'


# In[ ]:


if __name__ == "__main__":
    q = definite_integral_by_substitution()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=200)


# In[28]:


if __name__ == "__main__":
    pass
    #qz.save('definite_integral_by_substitution.xml')


# ## IntegrableByParts class for the following quizzes

# In[39]:


class IntegrableByParts():
    _const = sympy.Symbol('c') # non-zero integer
    _base = sympy.Symbol('a') # non-one positive integer
    _n = sympy.Symbol('n') # positive integer
    _x = sympy.Symbol('x', real=True)
    # in sympy, there is no way to represent log_a(x) without expanding.....
    # format: [f, F, incorrect Fs]
    _function_types = ['constant', 'monomial', 'rational', 'sine', 'cosine', 'tangent', 
                       'natural_exponent', 'general_exponent'] #, 'natural_logarithm']
    _function_defs = dict()
    _function_defs['constant'] = [_const, _const*_x, [_const, 0, 1]]
    _function_defs['monomial'] = [_x**_n, 1/(_n+1)*_x**(_n+1), [_x**_n, (_n)*_x**(_n-1)]]
    _function_defs['rational'] = [_x**(-_base), 1/(-_base+1)*_x**(-_base+1), [_x**(-_base), (-_base)*_x**(-_base-1)]]
    _function_defs['sine'] = [sympy.sin(_x), -sympy.cos(_x), [sympy.sin(_x), sympy.cos(_x)]]
    _function_defs['cosine'] = [sympy.cos(_x), sympy.sin(_x), [sympy.cos(_x), -sympy.sin(_x)]]
    _function_defs['tangent'] = [1/sympy.cos(_x)**2, sympy.tan(_x), [1/sympy.cos(_x), 1/sympy.sin(_x)**2]]
    _function_defs['natural_exponent'] = [sympy.exp(_x), sympy.exp(_x), [sympy.exp(_x-1), sympy.exp(_x+1)]]
    _function_defs['general_exponent'] = [_base**_x, 1/(sympy.log(_base))*_base**_x, [_base**_x, 1/(sympy.log(_base))*_base**(_x+1)]]
    _function_defs['natural_logarithm'] = [1/_x, sympy.log(abs(_x)), [1/_x**2, -1/_x**2]]
    # func = [scalar*linearity]+
    # linearity = elemental_func | elemental_func + constant
    _incorrect_reasons = ['formula', 'sign', 'scalar', 'part']
    def __init__(self, emin=-3, emax=3, nmin=1, nmax=3, srate=0.25):
        # 生成する個々の係数等の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 生成する次数の範囲
        self.deg_min = nmin
        self.deg_max = nmax
        # 定数項の確率
        self.constant_term_ratio = srate
        # internals
        self.function = 0
        self.is_trigonometric = False
        self.is_interval_should_be_positive = False
        self.is_interval_should_be_less_pi_2 = False
        self.interval = []
    # func = [scalar*linearity]+
    def generate_function(self):
        while self.function == 0:
            _type1 = random.choice(self._function_types[1:])
            _func1 = self._generate_linearity(_type1)
            if _type1 in ['monomial', 'rational']:
                _type2 = random.choice(self._function_types[3:])
            elif _type1 in ['natural_exponent', 'general_exponent']:
                _type2 = random.choice(self._function_types[3:6])
            else:
                _type2 = random.choice(self._function_types[1:])
            _func2 = self._generate_linearity(_type2)
            self.function = ['multiplication', _func1, _func2]
    # linearity = elemental_func | elemental_func + constant
    def _generate_linearity(self, _type=None):
        _scalar = nonzero_randint(self.elem_min, self.elem_max)
        _func1 = ['scalar', _scalar, self._generate_elemental_func(_type)]
        if random.random() >= self.constant_term_ratio:
            return _func1
        else:
            _func = ['summation', _func1]
            _func.append(self._generate_elemental_func('constant'))
        return _func
    def _mysubs(self, target, var, repl):
        if len(sympy.sympify(target).atoms(sympy.Symbol)) == 0:
            return target
        elif len(var) == 1:
            return target.subs(var[0], repl[0])
        else:
            return self._mysubs(target.subs(var[0], repl[0]), var[1:], repl[1:])
    def _generate_elemental_func(self, _type=None):
        if _type is None:
            _type = random.choice(self._function_types[1:])
        if _type in ['sine', 'cosine', 'tangent']:
            self.is_trigonometric = True
        if _type in ['tangent']:
            self.is_interval_should_be_less_pi_2 = True
        if _type in ['monomial', 'rational', 'natural_logarithm']:
            self.is_interval_should_be_positive = True            
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
        return sympy.diff(self._get_antiderivative(_func), self._x)
    def get_function(self):
        if self.function == 0:
            self.generate_function()
        return sympy.diff(self.get_antiderivative(), self._x)
    def generate_interval(self):
        if self.is_trigonometric:
            _candidates = [-sympy.pi, -sympy.pi/2, -sympy.pi/3, -sympy.pi/4, 0, sympy.pi/4, sympy.pi/3, sympy.pi/2, sympy.pi]
        else:
            _candidates = [sympy.Integer(_elem) for _elem in range(self.elem_min,self.elem_max+1)]
        if self.is_interval_should_be_less_pi_2:
            _candidates = [_elem for _elem in _candidates if abs(_elem) < sympy.pi/2]
        if self.is_interval_should_be_positive:
            _candidates = [_elem for _elem in _candidates if _elem > 0]
        self.interval = sorted(random.sample(_candidates,2))
    def _get_definite_integral(self, _func, _interval=None):
        if _interval is None:
            _interval = self.interval
        if isinstance(_func, list):
            _indefinite_integral = self._get_antiderivative(_func)
        else:
            _indefinite_integral = _func
        _fa = self._mysubs(_indefinite_integral, [self._x], [_interval[0]])        
        _fb = self._mysubs(_indefinite_integral, [self._x], [_interval[1]])
        return _fb - _fa
    def get_interval(self):
        if self.function == 0:
            self.generate_function()
        if len(self.interval) == 0:
            self.generate_interval()
        return self.interval        
    def get_definite_integral(self):
        if self.function == 0:
            self.generate_function()
        if len(self.interval) == 0:
            self.generate_interval()
        return self._get_definite_integral(self.function)
    def _get_antiderivative(self, _func):
        _recursive_call = self._get_antiderivative
        # ['summation', 'scalar']  
        if _func[0] == 'summation':
            _summand = 0
            for _f in _func[1:]:
                _summand = _summand + _recursive_call(_f)
            return _summand
        elif _func[0] == 'multiplication':
            return _recursive_call(_func[1]) * _recursive_call(_func[2])
        elif _func[0] == 'scalar':
            return _func[1] * _recursive_call(_func[2])
        else:
            return _func[1]
    def get_antiderivative(self):
        if self.function == 0:
            self.generate_function()
        return self._get_antiderivative(self.function)
    def _get_incorrect_antiderivatives_by_formula(self, _func):
        _recursive_call = self._get_incorrect_antiderivatives_by_formula
        if _func[0] == 'summation':
            _summand = [0]
            for _f in _func[1:]:
                _dfs = _recursive_call(_f)
                _new_summand = []
                for _df in _dfs:
                    _new_summand = _new_summand + [ _s + _df for _s in _summand]
                _summand = _new_summand
            return _summand
        elif _func[0] == 'multiplication':
            _funcs = []
            for _df1 in _recursive_call(_func[1]):
                for _df2 in _recursive_call(_func[2]):
                    _funcs.append(_df1 * self._get_function(_func[2]) + self._get_function(_func[1]) * _df2)
            return _funcs
        elif _func[0] == 'scalar':
            return [_func[1] * _f for _f in _recursive_call(_func[2])]
        else:
            return _func[2]
    def _get_incorrect_antiderivatives_by_sign(self, _func):
        _recursive_call = self._get_incorrect_antiderivatives_by_sign
        if _func[0] == 'summation':
            _summand = [0]
            for _f in _func[1:]:
                _dfs = _recursive_call(_f)
                _new_summand = []
                for _df in _dfs:
                    _new_summand = _new_summand + [ _s + _df for _s in _summand]
                _summand = _new_summand
            return _summand
        elif _func[0] == 'multiplication':
            _funcs = []
            for _df1 in _recursive_call(_func[1]):
                for _df2 in _recursive_call(_func[2]):
                    _funcs.append(_df1 * self._get_function(_func[2]) - self._get_function(_func[1]) * _df2)
            return _funcs
        elif _func[0] == 'scalar':
            return [_func[1] * _f for _f in _recursive_call(_func[2])]
        else:
            return [-_func[1]]
    def _get_incorrect_antiderivatives_by_scalar(self, _func):
        _recursive_call = self._get_incorrect_antiderivatives_by_scalar
        if _func[0] == 'summation':
            _summand = [0]
            for _f in _func[1:]:
                _dfs = _recursive_call(_f)
                _new_summand = []
                for _df in _dfs:
                    _new_summand = _new_summand + [ _s + _df for _s in _summand]
                _summand = _new_summand
            return _summand
        elif _func[0] == 'multiplication':
            _funcs = []
            for _df1 in _recursive_call(_func[1]):
                for _df2 in _recursive_call(_func[2]):
                    _funcs.append(_df1 * self._get_function(_func[2]) + self._get_function(_func[1]) * _df2)
            return _funcs
        elif _func[0] == 'scalar':
            return [_f for _f in _recursive_call(_func[2])]
        else:
            return [_func[1]]
    def _get_incorrect_antiderivatives_by_part(self, _func):
        _recursive_call = self._get_incorrect_antiderivatives_by_part
        # ['summation', 'scalar']  
        if _func[0] == 'summation':
            _funcs = []
            for _f in _func[1:]:
                _funcs = _funcs + _recursive_call(_f)
            return _funcs
        elif _func[0] == 'multiplication':
            _funcs = []
            for _df1 in _recursive_call(_func[1]):
                for _df2 in _recursive_call(_func[2]):
                    _funcs.append(_df1 * self._get_function(_func[2]) + self._get_function(_func[1]) * _df2)
            return _funcs
        elif _func[0] == 'scalar':
            return [_func[1] * _f for _f in _recursive_call(_func[2])]
        else:
            return [_func[1]]
    def _not_same_check(self, value):
        if not value.is_real:
            return False
        if value > 1.0e-4:
            return True
        return False
    def _incorrect_antiderivarives_only(self, _dfs):
        _funcs = []
        _correct_df = self.get_antiderivative()
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
    def get_incorrect_antiderivatives(self, _type=None):
        if self.function == 0:
            self.generate_function()
        if _type is None:
            _type = random.choice(self._incorrect_reasons)
        if _type == 'formula':
            _dfs = self._get_incorrect_antiderivatives_by_formula(self.function)
        elif _type == 'sign':
            _dfs = self._get_incorrect_antiderivatives_by_sign(self.function)
        elif _type == 'scalar':
            _dfs = self._get_incorrect_antiderivatives_by_scalar(self.function)
        elif _type == 'part':
            _dfs = self._get_incorrect_antiderivatives_by_part(self.function)
        return self._incorrect_antiderivarives_only(_dfs)
    @timeout_decorator.timeout(1)
    def _get_definite_integral_timeout(self, _df):
        return self._get_definite_integral(_df)
    def _incorrect_definite_integrals_only(self, _dfs):
        _dints = []
        _correct_dint = self.get_definite_integral()
        for _df in _dfs:
            try:
                _dint = self._get_definite_integral_timeout(_df)
            except:
                continue
            if _dint in _dints:
                continue
            if len(_dint.atoms(sympy.Symbol)) != 0:
                continue
            if self._not_same_check(abs(sympy.N(_correct_dint - _dint))):
                _dints.append(_dint)
        return _dints
    def get_incorrect_definite_integrals(self, _type=None):
        if self.function == 0:
            self.generate_function()
        if len(self.interval) == 0:
            self.generate_interval()
        if _type is None:
            _type = random.choice(self._incorrect_reasons)
        if _type == 'formula':
            _dfs = self._get_incorrect_antiderivatives_by_formula(self.function)
        elif _type == 'sign':
            _dfs = self._get_incorrect_antiderivatives_by_sign(self.function)
        elif _type == 'scalar':
            _dfs = self._get_incorrect_antiderivatives_by_scalar(self.function)
        elif _type == 'part':
            _dfs = self._get_incorrect_antiderivatives_by_part(self.function)
        return self._incorrect_definite_integrals_only(_dfs)


# In[40]:


if __name__ == "__main__":
    antidf = IntegrableByParts()
    antidf.generate_function()
    antidf.generate_interval()
    display(antidf.get_function())
    display(antidf.get_antiderivative())
    display(antidf.get_incorrect_antiderivatives())
    display(antidf.get_interval())
    display(antidf.get_definite_integral())
    display(antidf.get_incorrect_definite_integrals())


# ## antiderivative by parts

# In[41]:


class antiderivative_by_parts(core.Question):
    name = '部分積分に基づく不定積分の計算（積の微分法の逆）'
    def __init__(self, emin=-2, emax=2, nmin=1, nmax=2, srate=0.25):
        # 生成する個々の係数等の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 生成する次数の範囲
        self.deg_min = nmin
        self.deg_max = nmax
        # コンスタントの割合
        self.constant_rate = srate
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='部分積分に基づく不定積分の計算', quiz_number=_quiz_number)
        antidf = IntegrableByParts(emin=self.elem_min, emax=self.elem_max, nmin=self.deg_min, nmax=self.deg_max, srate=self.constant_rate)        
        _func = antidf.get_function()
        quiz.quiz_identifier = hash(str(_func))
        # 正答の選択肢の生成
        _antidf = antidf.get_antiderivative()
        quiz.data = [_func, _antidf, antidf]
        ans = { 'fraction': 100, 'data': _antidf }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_func, _antidf, antidf] = quiz.data
        ans['feedback'] = r'基本的な関数の不定積分の公式を確認してください。'
        for _incorrect in antidf.get_incorrect_antiderivatives('formula'):
            ans['data'] = _incorrect
            answers.append(dict(ans))
        ans['feedback'] = r'不定積分の線形性に基づいて計算してください。定数倍や加減算を忘れている可能性があります。'
        for _incorrect in antidf.get_incorrect_antiderivatives('scalar'):
            ans['data'] = _incorrect
            answers.append(dict(ans))
        for _incorrect in antidf.get_incorrect_antiderivatives('part'):
            ans['data'] = _incorrect
            answers.append(dict(ans))
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_func, _antidf, antidf] = quiz.data
        _text = r'次の不定積分を求めた結果を選択してください。なお，\( C \)を積分定数とします。'
        _text += r'<br />\( \displaystyle\int \left(' + sympy.latex(_func, order='lex') + r'\right) dx \)'
        return _text
    def answer_text(self, ans):
        return r'\( ' +  sympy.latex(ans['data'], order='lex') + r' +C \)'


# In[ ]:


if __name__ == "__main__":
    q = antiderivative_by_parts()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=200)


# In[43]:


if __name__ == "__main__":
    pass
    #qz.save('antiderivative_by_parts.xml')


# ## definite integral by parts

# In[45]:


class definite_integral_by_parts(core.Question):
    name = '部分積分に基づく定積分の計算（積の微分法の逆）'
    def __init__(self, emin=-2, emax=2, nmin=1, nmax=2, srate=0.25):
        # 生成する個々の係数等の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 生成する次数の範囲
        self.deg_min = nmin
        self.deg_max = nmax
        # コンスタントの割合
        self.constant_rate = srate
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='部分積分に基づく定積分の計算', quiz_number=_quiz_number)
        antidf = IntegrableByParts(emin=self.elem_min, emax=self.elem_max, nmin=self.deg_min, nmax=self.deg_max, srate=self.constant_rate)        
        _func = antidf.get_function()
        quiz.quiz_identifier = hash(str(_func))
        # 正答の選択肢の生成
        _dint = antidf.get_definite_integral()
        _interval = antidf.get_interval()
        quiz.data = [_func, _dint, _interval, antidf]
        ans = { 'fraction': 100, 'data': _dint }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_func, _dint, _interval, antidf] = quiz.data
        if -_dint != _dint:
            ans['feedback'] = r'定積分の区間の端点における原始関数の値の引き算の向きを確認しましょう。'
            ans['data'] = -_dint
            answers.append(dict(ans))
        ans['feedback'] = r'基本的な関数の不定積分の公式を確認してください。それを用いて定積分を求めます。'
        for _incorrect in antidf.get_incorrect_definite_integrals('formula'):
            ans['data'] = _incorrect
            answers.append(dict(ans))
            if _incorrect != _dint:
                ans['data'] = -_incorrect
                answers.append(dict(ans))
        ans['feedback'] = r'不定積分の線形性に基づいて計算してください。定数倍や加減算を忘れている可能性があります。それを用いて定積分を求めます。'
        for _incorrect in antidf.get_incorrect_definite_integrals('scalar'):
            ans['data'] = _incorrect
            answers.append(dict(ans))
            if _incorrect != _dint:
                ans['data'] = -_incorrect
                answers.append(dict(ans))
        for _incorrect in antidf.get_incorrect_definite_integrals('part'):
            ans['data'] = _incorrect
            answers.append(dict(ans))
            if _incorrect != _dint:
                ans['data'] = -_incorrect
                answers.append(dict(ans))
        answers = common.answer_union(answers)
        while len(answers) < size:
            ans['feedback'] = r'原始関数を求め，公式に基づき計算を行ってください。'
            _incorrect = _dint + nonzero_randint(self.elem_min, self.elem_max)
            ans['data'] = _incorrect
            answers.append(dict(ans))            
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_func, _dint, _interval, antidf] = quiz.data
        _text = r'次の定積分を求めた結果を選択してください。'
        _text += r'<br />\( \displaystyle\int_{' + sympy.latex(_interval[0]) + r'}^{' + sympy.latex(_interval[1]) + r'} \left(' + sympy.latex(_func, order='lex') + r'\right) dx \)'
        return _text
    def answer_text(self, ans):
        return r'\( ' +  sympy.latex(ans['data'], order='lex') + r' \)'


# In[ ]:


if __name__ == "__main__":
    q = definite_integral_by_parts()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=200)


# In[47]:


if __name__ == "__main__":
    pass
    #qz.save('definite_integral_by_parts.xml')


# ## Simple Differentiable Function class for the following quizzes

# In[35]:


class SimpleDifferentiableFunction():
    """
    return a function whose signs are same at x=0,1/100,2/100,...,1.
    """
    _const = sympy.Symbol('c') # non-zero integer
    _base = sympy.Symbol('a') # non-one positive integer
    _n = sympy.Symbol('n') # positive integer
    _x = sympy.Symbol('x')
    # in sympy, there is no way to represent log_a(x) without expanding.....
    # format: [f, df, incorrect dfs]
    _function_types = ['constant', 'monomial', 'sine', 'cosine', 'natural_exponent', 'general_exponent']
    _function_defs = dict()
    _function_defs['constant'] = [_const, 0, [_const, 1]]
    _function_defs['monomial'] = [_x**_n, _n*_x**(_n-1), [_x**_n, _n*_x**_n, (_n-1)*_x**(_n-1)]]
    _function_defs['sine'] = [sympy.sin(_x), sympy.cos(_x), [sympy.sin(_x), -sympy.cos(_x)]]
    _function_defs['cosine'] = [sympy.cos(_x), -sympy.sin(_x), [sympy.cos(_x), sympy.sin(_x)]]
    _function_defs['natural_exponent'] = [sympy.exp(_x), sympy.exp(_x), [sympy.exp(_x-1)]]
    _function_defs['general_exponent'] = [_base**_x, sympy.log(_base)*_base**_x, [_base**_x, _base**(_x-1)]]
    # func = [linearity_mono]+
    # linearity_mono = scalar*multi | scalar*quot | scalar*composite | scalar*elemental_func
    # composite|multi*quot = linearity @@ linearity
    # linearity = elemental_func | elemental_func + constant
    _expression_types = ['exponential', 'summation', 'scalar', 'composition', 'multiplication', 'quotient']
    _incorrect_reasons = ['formula', 'sign', 'composition', 'scalar', 'part']
    def __init__(self, emin=-3, emax=3, nmin=1, nmax=3, lmin=1, lmax=2, crate=0.25, mrate=0.25, qrate=0.25, srate=0.25):
        # 生成する個々の係数等の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 生成する次数の範囲
        self.deg_min = nmin
        self.deg_max = nmax
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
        # internals
        self.function = 0
    # func = linearity_mono**linearity_mono
    def generate_function(self):
        while True:
            _n = random.randint(self.linearity_min, self.linearity_max)
            _func1 = self._generate_linearity_mono()
            if _n == 1:
                _func = _func1
            else:
                _func = ['summation', _func1]
                for _i in range(_n-1):
                    _func.append(self._generate_linearity_mono())
            _f = self._get_function(_func)
            _slist = [sympy.sign(_f.subs(self._x, sympy.Rational(_i,100))) for _i in range(101)]
            _is_same = True
            for _s in _slist:
                if _s != _slist[0]:
                    _is_same = False
                    break
            if _is_same:
                break            
        self.function = _func
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
        return self._get_derivative(self.function)
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


# In[40]:


if __name__ == "__main__":
    df = SimpleDifferentiableFunction()
    df.generate_function()
    display(df.get_function())
    display(df.get_derivative())


# ## applications of integration

# In[57]:


class applications_of_integration(core.Question):
    name = '定積分の応用（体積，曲線の長さ，側面積）'
    qtype = ['volume', 'rvolume', 'length', 'area']
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
        quiz = core.Quiz(name='定積分の応用の公式利用', quiz_number=_quiz_number)
        sdf = SimpleDifferentiableFunction(emin=self.elem_min, emax=self.elem_max, nmin=self.deg_min, nmax=self.deg_max, 
                                           lmin=self.func_min, lmax=self.func_max, crate=0.5, mrate=0.0, qrate=0.0, srate=0.5)
        _func = sdf.get_function()
        _dfunc = sdf.get_derivative()
        _qtype = random.choice(self.qtype)        
        quiz.quiz_identifier = hash(str(_func) + str(_qtype))
        # 正答の選択肢の生成
        if _qtype == 'volume':
            _target_func = _func
        elif _qtype == 'rvolume':
            _target_func = _func**2
        elif _qtype == 'length':
            _target_func = sympy.sqrt(sympy.Integer(1)+_dfunc**2)
        else: # 'area'
            _target_func = _func*sympy.sqrt(sympy.Integer(1)+_dfunc**2)
        quiz.data = [_func, _dfunc, _qtype, sdf, _target_func]
        ans = { 'fraction': 100, 'data': [_qtype, _target_func] }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_func, _dfunc, _qtype, sdf, _target_func] = quiz.data
        _incorrect_dfuncs = sdf.get_incorrect_derivatives()
        ans['feedback'] = r'体積，曲線の長さ，側面積のそれぞれの公式を確認してください。'
        if _qtype == 'volume':
            ans['data'] = ['rvolume', _target_func]
            answers.append(dict(ans))
            ans['data'] = ['area', _target_func]
            answers.append(dict(ans))
            _incorrect_target_func = _func**2
            ans['data'] = ['volume', _incorrect_target_func]
            answers.append(dict(ans))
            ans['data'] = ['rvolume', _incorrect_target_func]
            answers.append(dict(ans))
            ans['data'] = ['area', _incorrect_target_func]
            answers.append(dict(ans))
            _incorrect_target_func = sympy.sqrt(sympy.Integer(1)+_dfunc**2)
            ans['data'] = ['volume', _incorrect_target_func]
            answers.append(dict(ans))
            ans['data'] = ['rvolume', _incorrect_target_func]
            answers.append(dict(ans))
            ans['data'] = ['area', _incorrect_target_func]
            answers.append(dict(ans))
            _incorrect_target_func = _func*sympy.sqrt(sympy.Integer(1)+_dfunc**2)  
            ans['data'] = ['volume', _incorrect_target_func]
            answers.append(dict(ans))
            ans['data'] = ['rvolume', _incorrect_target_func]
            answers.append(dict(ans))
            ans['data'] = ['area', _incorrect_target_func]
            answers.append(dict(ans))          
            if len(_incorrect_dfuncs) > 0: 
                ans['feedback'] = r'体積，曲線の長さ，側面積のそれぞれの公式を確認してください。導関数の計算もおかしいです。'               
                _incorrect_target_func = sympy.sqrt(sympy.Integer(1)+_incorrect_dfuncs[0]**2)
                ans['data'] = ['volume', _incorrect_target_func]
                answers.append(dict(ans))
                ans['data'] = ['rvolume', _incorrect_target_func]
                answers.append(dict(ans))
                ans['data'] = ['area', _incorrect_target_func]
                answers.append(dict(ans))
                _incorrect_target_func = _func*sympy.sqrt(sympy.Integer(1)+_incorrect_dfuncs[0]**2)  
                ans['data'] = ['volume', _incorrect_target_func]
                answers.append(dict(ans))
                ans['data'] = ['rvolume', _incorrect_target_func]
                answers.append(dict(ans))
                ans['data'] = ['area', _incorrect_target_func]
                answers.append(dict(ans))          
        elif _qtype == 'length':
            ans['data'] = ['rvolume', _target_func]
            answers.append(dict(ans))
            ans['data'] = ['area', _target_func]
            answers.append(dict(ans))
            _incorrect_target_func = _func
            ans['data'] = ['length', _incorrect_target_func]
            answers.append(dict(ans))
            ans['data'] = ['rvolume', _incorrect_target_func]
            answers.append(dict(ans))
            ans['data'] = ['area', _incorrect_target_func]
            answers.append(dict(ans))          
            _incorrect_target_func = _func**2
            ans['data'] = ['length', _incorrect_target_func]
            answers.append(dict(ans))
            ans['data'] = ['rvolume', _incorrect_target_func]
            answers.append(dict(ans))
            ans['data'] = ['area', _incorrect_target_func]
            answers.append(dict(ans))          
            _incorrect_target_func = _func*sympy.sqrt(sympy.Integer(1)+_dfunc**2)
            ans['data'] = ['length', _incorrect_target_func]
            answers.append(dict(ans))
            ans['data'] = ['rvolume', _incorrect_target_func]
            answers.append(dict(ans))
            ans['data'] = ['area', _incorrect_target_func]
            answers.append(dict(ans))        
            if len(_incorrect_dfuncs) > 0:         
                ans['feedback'] = r'体積，曲線の長さ，側面積のそれぞれの公式を確認してください。導関数の計算もおかしいです。'               
                _incorrect_target_func = _func*sympy.sqrt(sympy.Integer(1)+_incorrect_dfuncs[0]**2)
                ans['data'] = ['length', _incorrect_target_func]
                answers.append(dict(ans))
                ans['data'] = ['rvolume', _incorrect_target_func]
                answers.append(dict(ans))
                ans['data'] = ['area', _incorrect_target_func]
                answers.append(dict(ans))               
        elif _qtype == 'rvolume':
            ans['data'] = ['volume', _target_func]
            answers.append(dict(ans))
            ans['data'] = ['area', _target_func]
            answers.append(dict(ans))
            _incorrect_target_func = _func
            ans['data'] = ['rvolume', _incorrect_target_func]
            answers.append(dict(ans))
            ans['data'] = ['volume', _incorrect_target_func]
            answers.append(dict(ans))
            ans['data'] = ['area', _incorrect_target_func]
            answers.append(dict(ans))          
            _incorrect_target_func = sympy.sqrt(sympy.Integer(1)+_dfunc**2)
            ans['data'] = ['rvolume', _incorrect_target_func]
            answers.append(dict(ans))
            ans['data'] = ['volume', _incorrect_target_func]
            answers.append(dict(ans))
            ans['data'] = ['area', _incorrect_target_func]
            answers.append(dict(ans))         
            _incorrect_target_func = _func*sympy.sqrt(sympy.Integer(1)+_dfunc**2)
            ans['data'] = ['rvolume', _incorrect_target_func]
            answers.append(dict(ans))
            ans['data'] = ['volume', _incorrect_target_func]
            answers.append(dict(ans))
            ans['data'] = ['area', _incorrect_target_func]
            answers.append(dict(ans))        
            if len(_incorrect_dfuncs) > 0:   
                ans['feedback'] = r'体積，曲線の長さ，側面積のそれぞれの公式を確認してください。導関数の計算もおかしいです。'               
                _incorrect_target_func = sympy.sqrt(sympy.Integer(1)+_incorrect_dfuncs[0]**2)
                ans['data'] = ['rvolume', _incorrect_target_func]
                answers.append(dict(ans))
                ans['data'] = ['volume', _incorrect_target_func]
                answers.append(dict(ans))
                ans['data'] = ['area', _incorrect_target_func]
                answers.append(dict(ans))         
                _incorrect_target_func = _func*sympy.sqrt(sympy.Integer(1)+_incorrect_dfuncs[0]**2)
                ans['data'] = ['rvolume', _incorrect_target_func]
                answers.append(dict(ans))
                ans['data'] = ['volume', _incorrect_target_func]
                answers.append(dict(ans))
                ans['data'] = ['area', _incorrect_target_func]
                answers.append(dict(ans))    
        else: # 'area'
            ans['data'] = ['volume', _target_func]
            answers.append(dict(ans))
            ans['data'] = ['rvolume', _target_func]
            answers.append(dict(ans))
            _incorrect_target_func = _func
            ans['data'] = ['area', _incorrect_target_func]
            answers.append(dict(ans))
            ans['data'] = ['volume', _incorrect_target_func]
            answers.append(dict(ans))
            ans['data'] = ['rvolume', _incorrect_target_func]
            answers.append(dict(ans))         
            _incorrect_target_func = _func**2
            ans['data'] = ['area', _incorrect_target_func]
            answers.append(dict(ans))
            ans['data'] = ['volume', _incorrect_target_func]
            answers.append(dict(ans))
            ans['data'] = ['rvolume', _incorrect_target_func]
            answers.append(dict(ans))         
            _incorrect_target_func = sympy.sqrt(sympy.Integer(1)+_dfunc**2)
            ans['data'] = ['area', _incorrect_target_func]
            answers.append(dict(ans))
            ans['data'] = ['volume', _incorrect_target_func]
            answers.append(dict(ans))
            ans['data'] = ['rvolume', _incorrect_target_func]
            answers.append(dict(ans))            
            if len(_incorrect_dfuncs) > 0:  
                ans['feedback'] = r'体積，曲線の長さ，側面積のそれぞれの公式を確認してください。導関数の計算もおかしいです。'               
                _incorrect_target_func = sympy.sqrt(sympy.Integer(1)+_incorrect_dfuncs[0]**2)
                ans['data'] = ['area', _incorrect_target_func]
                answers.append(dict(ans))
                ans['data'] = ['volume', _incorrect_target_func]
                answers.append(dict(ans))
                ans['data'] = ['rvolume', _incorrect_target_func]
                answers.append(dict(ans))
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_func, _dfunc, _qtype, sdf, _target_func] = quiz.data
        _text = r'\( f(x) \)を次の関数とするとき，'
        if _qtype == 'volume':
            _text += r'\( x=0 \)から\( x=1 \)までの切断面の面積が\( f(x) \)の空間図形の体積の計算式を選んでください。'
        elif _qtype == 'rvolume':
            _text += r'\( y=f(x) \)のグラフと直線\( x=0 \)と\( x=1 \)で囲まれた図形を\( x \)軸を中心に1回転させてできる空間図形の体積の計算式を選んでください。'
        elif _qtype == 'length':
            _text += r'\( y=f(x) \)のグラフの\( x=0 \)から\( x=1 \)までの道のりの計算式を選んでください。'
        else: # 'area'
            _text += r'\( y=f(x) \)のグラフと直線\( x=0 \)と\( x=1 \)で囲まれた図形を\( x \)軸を中心に1回転させてできる空間図形の側面積の計算式を選んでください。'
        _text += r'<br />\( \displaystyle f(x)=' + sympy.latex(_func, order='lex') + r' \)'
        return _text
    def answer_text(self, ans):
        [_qtype, _target_func] = ans['data']
        _text = r'\( '
        if _qtype == 'volume':
            _text += r'\displaystyle\int_0^1 ' + sympy.latex(_target_func, order='lex') + r' dx'
        elif _qtype == 'rvolume':
            _text += r'\displaystyle\pi\int_0^1 ' + sympy.latex(_target_func, order='lex') + r' dx'
        elif _qtype == 'length':
            _text += r'\displaystyle\int_0^1 ' + sympy.latex(_target_func, order='lex') + r' dx'
        else: # 'area'
            _text += r'\displaystyle 2\pi\int_0^1 ' + sympy.latex(_target_func, order='lex') + r' dx'
        return _text + r' \)'


# In[58]:


if __name__ == "__main__":
    q = applications_of_integration()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=25)


# In[59]:


if __name__ == "__main__":
    pass
    #qz.save('applications_of_integration.xml')


# In[ ]:





# ## dummy

# In[ ]:





# # All the questions

# In[ ]:


questions_str = ['antiderivative_by_linearity', 'definite_integral_by_linearity', 
                 'antiderivative_by_substitution', 'definite_integral_by_substitution', 
                 'antiderivative_by_parts', 'definite_integral_by_parts', 'applications_of_integration']
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




