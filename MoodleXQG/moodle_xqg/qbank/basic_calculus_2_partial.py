#!/usr/bin/env python
# coding: utf-8

# Moodle XML Question Generator, (C) 2020 Kosaku Nagasaka (Kobe University)

# ## モジュールファイルの生成

# In[1]:


if __name__ == "__main__":
    import subprocess
    subprocess.run(['jupyter', 'nbconvert', '--to', 'python', '_basic_calculus_2_partial.ipynb','--output','basic_calculus_2_partial.py'])


# # Basic Calculus 2 partial (differencial calculus for bivariate function)

# In[2]:


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


# In[3]:


import sympy
import random
import IPython
import itertools
import copy


# ## minor helpers

# In[4]:


def nonzero_randint(imin, imax):
    if 0 < imin or imax < 0:
        return random.randint(imin, imax)
    else:
        return random.choice(list(set(range(imin,imax+1)).difference(set([0]))))


# In[5]:


def nonone_randpint(imin, imax):
    if 1 < imin:
        return random.randint(imin, imax)
    else:
        return abs(random.choice(list(set(range(imin,imax+1)).difference(set([0,-1,1])))))


# In[6]:


def flatten_list(alist):
    rlist = []
    for lis in alist:
        rlist = rlist + lis
    return rlist


# In[7]:


def flatten_list_all(alist):
    if type(alist) is not list:
        return [alist]
    rlist = []
    for lis in alist:
        rlist = rlist + flatten_list_all(lis)
    return rlist


# ## limit of bivariate polynomial

# In[114]:


class limit_of_bivariate_polynomial(core.Question):
    name = '2変数多項式の極限（原則代入で求まる）'
    def __init__(self, emin=-3, emax=3, amin=-1, amax=1, nmin=0, nmax=3, tmin=2, tmax=4):
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 極限を求めるxの値の範囲
        self.a_min = amin
        self.a_max = amax
        # 次数の範囲
        self.deg_min = nmin
        self.deg_max = nmax
        # 単項式の個数の範囲
        self.term_min = tmin
        self.term_max = tmax
    def _function_generate(self, _varX, _varY):
        _poly = 0
        _num_terms = random.randint(self.term_min, self.term_max)
        _exps = flatten_list([[[_i,_j] for _j in range(self.deg_min, self.deg_max+1)] for _i in range(self.deg_min, self.deg_max+1)])
        _terms = random.sample(_exps,k=_num_terms)        
        for _es in _terms:
            _poly += nonzero_randint(self.elem_min, self.elem_max)*_varX**_es[0]*_varY**_es[1]
        return _poly
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='2変数多項式の極限', quiz_number=_quiz_number)
        _varX = sympy.Symbol('x')
        _varY = sympy.Symbol('y')
        _f = self._function_generate(_varX, _varY)
        _a = random.choice(list(range(self.a_min, self.a_max+1)))
        _b = random.choice(list(range(self.a_min, self.a_max+1)))
        _limit = _f.subs(_varX,_a).subs(_varY,_b)
        quiz.quiz_identifier = hash(str(_f) + str(_a) + str(_b) + str(_limit))
        # 正答の選択肢の生成
        quiz.data = [_varX, _varY, _f, _a, _b, _limit]
        ans = { 'fraction': 100, 'data': _limit }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_varX, _varY, _f, _a, _b, _limit] = quiz.data
        ans['feedback'] = r'多項式関数の極限は存在します。'
        ans['data'] = r'極限は存在しない'
        answers.append(dict(ans))
        ans['feedback'] = r'多項式関数の有限点での極限は発散しません。'
        ans['data'] = sympy.S.Infinity
        answers.append(dict(ans))
        ans['feedback'] = r'多項式関数の有限点での極限は発散しません。'
        ans['data'] = -sympy.S.Infinity
        answers.append(dict(ans))
        if _limit != 0:
            ans['feedback'] = r'何か計算ミスをしていないでしょうか。再度確認を行いましょう。'
            ans['data'] = 0
            answers.append(dict(ans))
            ans['feedback'] = r'何か計算ミスをしていないでしょうか。再度確認を行いましょう。'
            ans['data'] = -_limit
            answers.append(dict(ans))
        while True:
            _incorrect_a = random.choice(list(range(self.a_min-1, self.a_max+2)))
            _incorrect_b = random.choice(list(range(self.a_min-1, self.a_max+2)))
            _incorrect_limit = _f.subs(_varX,_incorrect_a).subs(_varY,_incorrect_b)
            if abs(_incorrect_limit) != abs(_limit):
                break
        ans['feedback'] = r'何か計算ミスをしていないでしょうか。再度確認を行いましょう。'
        ans['data'] = _incorrect_limit
        answers.append(dict(ans))
        ans['feedback'] = r'何か計算ミスをしていないでしょうか。再度確認を行いましょう。'
        ans['data'] = -_incorrect_limit
        answers.append(dict(ans))        
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_varX, _varY, _f, _a, _b, _limit] = quiz.data
        _text = r'次の極限を選択してください。<br />'
        _text += r'\( \lim_{(x,y)\rightarrow(' + sympy.latex(_a) + r',' + sympy.latex(_b) + r')}' + sympy.latex(_f, order='lex') + r' \)'
        return _text
    def answer_text(self, ans):
        if isinstance(ans['data'], str):
            return ans['data']
        else:
            return r'\( ' +  sympy.latex(ans['data'], order='lex') + r' \)'


# In[115]:


if __name__ == "__main__":
    q = limit_of_bivariate_polynomial()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=25)


# In[116]:


if __name__ == "__main__":
    pass
    #qz.save('limit_of_bivariate_polynomial.xml')


# ## limit of bivariate rational function approaches domain

# In[108]:


class limit_of_bivariate_rational_function_approaches_domain(core.Question):
    name = '2変数有理関数の極限（原則代入で求まる定義域への極限）'
    def __init__(self, emin=-3, emax=3, amin=-1, amax=1, nmin=0, nmax=3, tmin=1, tmax=3):
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 極限を求めるxの値の範囲
        self.a_min = amin
        self.a_max = amax
        # 次数の範囲
        self.deg_min = nmin
        self.deg_max = nmax
        # 単項式の個数の範囲
        self.term_min = tmin
        self.term_max = tmax
    def _function_generate(self, _varX, _varY, _a, _b):
        _num = 0
        _num_terms = random.randint(self.term_min, self.term_max)
        _exps = flatten_list([[[_i,_j] for _j in range(self.deg_min, self.deg_max+1)] for _i in range(self.deg_min, self.deg_max+1)])
        _terms = random.sample(_exps,k=_num_terms)        
        for _es in _terms:
            _num += nonzero_randint(self.elem_min, self.elem_max)*_varX**_es[0]*_varY**_es[1]
        _den = 0
        _num_terms = random.randint(self.term_min, self.term_max)
        _exps = flatten_list([[[_i,_j] for _j in range(self.deg_min, self.deg_max+1)] for _i in range(self.deg_min, self.deg_max+1)])
        _terms = random.sample(_exps,k=_num_terms)        
        for _es in _terms:
            _den += nonzero_randint(self.elem_min, self.elem_max)*_varX**_es[0]*_varY**_es[1]
        if sympy.degree(_den, _varX) + sympy.degree(_den, _varY) <= 0:
            return self._function_generate(_varX, _varY, _a, _b)
        elif sympy.degree(_num, _varX) + sympy.degree(_num, _varY) <= 0:
            return self._function_generate(_varX, _varY, _a, _b)
        elif _den.subs(_varX,_a).subs(_varY,_b) == 0:
            return self._function_generate(_varX, _varY, _a, _b)
        else:
            return _num/_den
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='2変数有理関数の極限（定義域）', quiz_number=_quiz_number)
        _varX = sympy.Symbol('x')
        _varY = sympy.Symbol('y')
        _a = random.choice(list(range(self.a_min, self.a_max+1)))
        _b = random.choice(list(range(self.a_min, self.a_max+1)))
        _f = self._function_generate(_varX, _varY, _a, _b)
        _limit = _f.subs(_varX,_a).subs(_varY,_b)
        quiz.quiz_identifier = hash(str(_f) + str(_a) + str(_b) + str(_limit))
        # 正答の選択肢の生成
        quiz.data = [_varX, _varY, _f, _a, _b, _limit]
        ans = { 'fraction': 100, 'data': _limit }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_varX, _varY, _f, _a, _b, _limit] = quiz.data
        ans['feedback'] = r'有理関数の極限は，分母の零点でなければ必ず存在します。'
        ans['data'] = r'極限は存在しない'
        answers.append(dict(ans))
        ans['feedback'] = r'有理関数の有限点での極限は，分母の零点でなければ発散しません。'
        ans['data'] = sympy.S.Infinity
        answers.append(dict(ans))
        ans['feedback'] = r'有理関数の有限点での極限は，分母の零点でなければ発散しません。'
        ans['data'] = -sympy.S.Infinity
        answers.append(dict(ans))
        if _limit != 0:
            ans['feedback'] = r'何か計算ミスをしていないでしょうか。再度確認を行いましょう。'
            ans['data'] = 0
            answers.append(dict(ans))
            ans['feedback'] = r'何か計算ミスをしていないでしょうか。再度確認を行いましょう。'
            ans['data'] = -_limit
            answers.append(dict(ans))
        _num_loop = 0
        while _num_loop < 10:
            _incorrect_a = random.choice(list(range(self.a_min-1, self.a_max+2)))
            _incorrect_b = random.choice(list(range(self.a_min-1, self.a_max+2)))
            _incorrect_limit = _f.subs(_varX,_incorrect_a).subs(_varY,_incorrect_b)
            if abs(_incorrect_limit) != abs(_limit) and _incorrect_limit.is_real:
                break
            else:
                _num_loop += 1
        if _num_loop < 10:
            ans['feedback'] = r'何か計算ミスをしていないでしょうか。再度確認を行いましょう。'
            ans['data'] = _incorrect_limit
            answers.append(dict(ans))
            ans['feedback'] = r'何か計算ミスをしていないでしょうか。再度確認を行いましょう。'
            ans['data'] = -_incorrect_limit
            answers.append(dict(ans))        
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_varX, _varY, _f, _a, _b, _limit] = quiz.data
        _text = r'次の極限を選択してください。<br />'
        _text += r'\( \lim_{(x,y)\rightarrow(' + sympy.latex(_a) + r',' + sympy.latex(_b) + r')}' + sympy.latex(_f, order='lex') + r' \)'
        return _text
    def answer_text(self, ans):
        if isinstance(ans['data'], str):
            return ans['data']
        else:
            return r'\( ' +  sympy.latex(ans['data'], order='lex') + r' \)'


# In[109]:


if __name__ == "__main__":
    q = limit_of_bivariate_rational_function_approaches_domain()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=25)


# In[110]:


if __name__ == "__main__":
    pass
    #qz.save('limit_of_bivariate_rational_function_approaches_domain.xml')


# ## limit of bivariate rational function approaches domain with common factor

# In[111]:


class limit_of_bivariate_rational_function_approaches_domain_with_common_factor(core.Question):
    name = '2変数有理関数の極限（約分後，原則代入で求まる定義域への極限）'
    def __init__(self, emin=-3, emax=3, amin=-1, amax=1, nmin=0, nmax=2, tmin=1, tmax=2):
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 極限を求めるxの値の範囲
        self.a_min = amin
        self.a_max = amax
        # 次数の範囲
        self.deg_min = nmin
        self.deg_max = nmax
        # 単項式の個数の範囲
        self.term_min = tmin
        self.term_max = tmax
    def _function_generate(self, _varX, _varY, _a, _b):
        _num = 0
        _num_terms = random.randint(self.term_min, self.term_max)
        _exps = flatten_list([[[_i,_j] for _j in range(self.deg_min, self.deg_max+1)] for _i in range(self.deg_min, self.deg_max+1)])
        _terms = random.sample(_exps,k=_num_terms)        
        for _es in _terms:
            _num += nonzero_randint(self.elem_min, self.elem_max)*_varX**_es[0]*_varY**_es[1]
        _den_nonzero = 0
        _num_terms = random.randint(self.term_min, self.term_max)
        _exps = flatten_list([[[_i,_j] for _j in range(self.deg_min, self.deg_max+1)] for _i in range(self.deg_min, self.deg_max+1)])
        _terms = random.sample(_exps,k=_num_terms)  
        for _es in _terms:
            _den_nonzero += nonzero_randint(self.elem_min, self.elem_max)*_varX**_es[0]*_varY**_es[1]
        if _a == 0 and _b == 0:
            _den_zero = (_varX - _a)*random.choice([1,-1]) + (_varY - _b)*random.choice([1,-1])
        elif _a != 0:
            _den_zero = (_varX - _a)*random.choice([1,-1])
        elif _b != 0:
            _den_zero = (_varY - _b)*random.choice([1,-1])
        if sympy.degree(_den_nonzero, _varX) + sympy.degree(_den_nonzero, _varY) <= 0:
            return self._function_generate(_varX, _varY, _a, _b)
        elif _den_nonzero.subs(_varX,_a).subs(_varY,_b) == 0:
            return self._function_generate(_varX, _varY, _a, _b)
        else:
            return sympy.expand(_num*_den_zero)/sympy.expand(_den_nonzero*_den_zero), (_num/_den_nonzero).subs(_varX,_a).subs(_varY,_b)
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='2変数有理関数の極限（約分後，定義域）', quiz_number=_quiz_number)
        _varX = sympy.Symbol('x')
        _varY = sympy.Symbol('y')
        _a = random.choice(list(range(self.a_min, self.a_max+1)))
        _b = random.choice(list(range(self.a_min, self.a_max+1)))
        _f,_limit = self._function_generate(_varX, _varY, _a, _b)
        quiz.quiz_identifier = hash(str(_f) + str(_a) + str(_b) + str(_limit))
        # 正答の選択肢の生成
        quiz.data = [_varX, _varY, _f, _a, _b, _limit]
        ans = { 'fraction': 100, 'data': _limit }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_varX, _varY, _f, _a, _b, _limit] = quiz.data
        ans['feedback'] = r'有理関数の極限は，分母の零点でなければ必ず存在します。'
        ans['data'] = r'極限は存在しない'
        answers.append(dict(ans))
        ans['feedback'] = r'有理関数の有限点での極限は，分母の零点でなければ発散しません。'
        ans['data'] = sympy.S.Infinity
        answers.append(dict(ans))
        ans['feedback'] = r'有理関数の有限点での極限は，分母の零点でなければ発散しません。'
        ans['data'] = -sympy.S.Infinity
        answers.append(dict(ans))
        if _limit != 0:
            ans['feedback'] = r'何か計算ミスをしていないでしょうか。再度確認を行いましょう。'
            ans['data'] = 0
            answers.append(dict(ans))
            ans['feedback'] = r'何か計算ミスをしていないでしょうか。再度確認を行いましょう。'
            ans['data'] = -_limit
            answers.append(dict(ans))
        for _i in range(10):
            _incorrect_a = random.choice(list(range(self.a_min-1, self.a_max+2)))
            _incorrect_b = random.choice(list(range(self.a_min-1, self.a_max+2)))
            _incorrect_limit = _f.subs(_varX,_incorrect_a).subs(_varY,_incorrect_b)
            if abs(_incorrect_limit) != abs(_limit) and _incorrect_limit.is_real:
                break
        else:
            _incorrect_limit = _limit + random.choice([-1,1])            
            while abs(_incorrect_limit) == abs(_limit):
                _incorrect_limit += random.choice([-1,1])
        ans['feedback'] = r'何か計算ミスをしていないでしょうか。再度確認を行いましょう。'
        ans['data'] = _incorrect_limit
        answers.append(dict(ans))
        ans['feedback'] = r'何か計算ミスをしていないでしょうか。再度確認を行いましょう。'
        ans['data'] = -_incorrect_limit
        answers.append(dict(ans))        
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_varX, _varY, _f, _a, _b, _limit] = quiz.data
        _text = r'次の極限を選択してください。<br />'
        _text += r'\( \lim_{(x,y)\rightarrow(' + sympy.latex(_a) + r',' + sympy.latex(_b) + r')}' + sympy.latex(_f, order='lex') + r' \)'
        return _text
    def answer_text(self, ans):
        if isinstance(ans['data'], str):
            return ans['data']
        else:
            return r'\( ' +  sympy.latex(ans['data'], order='lex') + r' \)'


# In[112]:


if __name__ == "__main__":
    q = limit_of_bivariate_rational_function_approaches_domain_with_common_factor()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=25)


# In[113]:


if __name__ == "__main__":
    pass
    #qz.save('limit_of_bivariate_rational_function_approaches_domain_with_common_factor.xml')


# ## limit of bivariate rational function approaches boundary

# In[98]:


class limit_of_bivariate_rational_function_approaches_boundary(core.Question):
    name = '2変数有理関数の極限（定義域の境界点への極限）'
    def __init__(self, emin=-3, emax=3, amin=-1, amax=1, nmin=1, nmax=3, tmin=1, tmax=2, lrate=0.5):
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 極限を求めるxの値の範囲
        self.a_min = amin
        self.a_max = amax
        # 次数の範囲
        self.deg_min = nmin
        self.deg_max = nmax
        # 単項式の個数の範囲
        self.term_min = tmin
        self.term_max = tmax
        # 極限が存在する確率
        self.limit_exist_ratio = lrate
    def _function_generate(self, _varX, _varY, _a, _b, _is_limit_exists=None):
        if _is_limit_exists is None:
            _is_limit_exists = True if random.random() <= self.limit_exist_ratio else False
        _deg_den = random.randint(self.deg_min, self.deg_max)
        if _is_limit_exists:
            _deg_num = _deg_den + 1
        else:
            _deg_num = _deg_den
        _num = 0
        _num_terms = random.randint(self.term_min, self.term_max)
        _exps = [[_i,_deg_num-_i] for _i in range(_deg_num+1)]
        if _num_terms < len(_exps):
            _terms = random.sample(_exps,k=_num_terms)
        else:
            _terms = _exps
        for _es in _terms:
            _num += nonzero_randint(self.elem_min, self.elem_max)*(_varX-_a)**_es[0]*(_varY-_b)**_es[1]
        _den = 0
        _num_terms = random.randint(self.term_min, self.term_max)
        _exps = [[_i,_deg_den-_i] for _i in range(_deg_den+1)]
        if _num_terms < len(_exps):
            _terms = random.sample(_exps,k=_num_terms)
        else:
            _terms = _exps
        for _es in _terms:
            _den += nonzero_randint(self.elem_min, self.elem_max)*(_varX-_a)**_es[0]*(_varY-_b)**_es[1]
        if sympy.degree(_num, _varX)*sympy.degree(_num, _varY)*sympy.degree(_den, _varX)*sympy.degree(_den, _varY) == 0:
            return self._function_generate(_varX, _varY, _a, _b, _is_limit_exists)
        if _is_limit_exists:
            return _num, _den, 0
        else:
            _varM = sympy.Symbol('m')
            _limit = sympy.expand(sympy.cancel((_num/_den).subs(_varX,_varM).subs(_varY,_varM)))
            if _limit.is_polynomial(_varM):
                if sympy.degree(_limit, _varM) <= 0:
                    return self._function_generate(_varX, _varY, _a, _b, _is_limit_exists)
            return _num, _den, r'極限は存在しない'
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='2変数有理関数の極限（境界点）', quiz_number=_quiz_number)
        _varX = sympy.Symbol('x')
        _varY = sympy.Symbol('y')
        _a = random.choice(list(range(self.a_min, self.a_max+1)))
        _b = random.choice(list(range(self.a_min, self.a_max+1)))
        _fn, _fd, _limit = self._function_generate(_varX, _varY, _a, _b)
        quiz.quiz_identifier = hash(str(_fn) + str(_fd) + str(_a) + str(_b) + str(_limit))
        # 正答の選択肢の生成
        quiz.data = [_varX, _varY, _fn, _fd, _a, _b, _limit]
        ans = { 'fraction': 100, 'data': _limit }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_varX, _varY, _fn, _fd, _a, _b, _limit] = quiz.data
        if isinstance(_limit, str):
            ans['feedback'] = r'\( y=mx \)などの変換を行い，本当に，近づけ方によらずに一定の値に近づくのかを確認しましょう。'
            ans['data'] = 0
            answers.append(dict(ans))
            ans['data'] = sympy.S.Infinity
            answers.append(dict(ans))
            ans['data'] = -sympy.S.Infinity
            answers.append(dict(ans))
            ans['data'] = 1
            answers.append(dict(ans))
            ans['data'] = -1
            answers.append(dict(ans))
        else:
            ans['feedback'] = r'極座標変換を行うなどして，挟み込みをしてみましょう。'
            ans['data'] = r'極限は存在しない'
            answers.append(dict(ans))
            ans['data'] = sympy.S.Infinity
            answers.append(dict(ans))
            ans['data'] = -sympy.S.Infinity
            answers.append(dict(ans))
            ans['data'] = 1
            answers.append(dict(ans))
            ans['data'] = -1
            answers.append(dict(ans))
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_varX, _varY, _fn, _fd, _a, _b, _limit] = quiz.data
        _text = r'次の極限を選択してください。<br />'
        _text += r'\( \lim_{(x,y)\rightarrow(' + sympy.latex(_a) + r',' + sympy.latex(_b) + r')}\frac{'
        _text += sympy.latex(_fn, order='lex') + r'}{' + sympy.latex(_fd, order='lex') + r'} \)'
        return _text
    def answer_text(self, ans):
        if isinstance(ans['data'], str):
            return ans['data']
        else:
            return r'\( ' +  sympy.latex(ans['data'], order='lex') + r' \)'


# In[99]:


if __name__ == "__main__":
    q = limit_of_bivariate_rational_function_approaches_boundary()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=25)


# In[100]:


if __name__ == "__main__":
    pass
    #qz.save('limit_of_bivariate_rational_function_approaches_boundary.xml')


# ## limit with sin(x+y) and log(1+x+y) over (x+y)

# In[117]:


class limit_with_sin_or_log_over_xpy(core.Question):
    name = '2変数有理関数の極限（分解でsinやlogの1変数に帰着）'
    def __init__(self, emin=-3, emax=3, amin=0, amax=0, nmin=0, nmax=1, tmin=1, tmax=2):
        # 生成する個々の数の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 極限を求めるxの値の範囲
        self.a_min = amin
        self.a_max = amax
        # 次数の範囲
        self.deg_min = nmin
        self.deg_max = nmax
        # 単項式の個数の範囲
        self.term_min = tmin
        self.term_max = tmax
        # 極限が存在する確率（境界点への極限計算の場合における）
        self.limit_exist_ratio = 0.5
    def _function_generate(self, _varX, _varY, _a, _b):
        _c = nonzero_randint(self.elem_min, self.elem_max)
        if random.random() <= 0.5:
            _pre_num = sympy.sin(_c*(_varX-_a) + _c*(_varY-_b))
            _pre_den = _c*(_varX-_a) + _c*(_varY-_b)
        else:
            _pre_num = sympy.log(1+_c*(_varX-_a) + _c*(_varY-_b))
            _pre_den = _c*(_varX-_a) + _c*(_varY-_b)
        _t = random.choice(['poly','ratd', 'ratf', 'ratb'])
        if _t == 'poly':
            _num, _den, _limit = self._function_generate_polynomial(_varX, _varY, _a, _b)
        elif _t == 'ratd':
            _num, _den, _limit = self._function_generate_rational_domain(_varX, _varY, _a, _b)
        elif _t == 'ratf':
            _num, _den, _limit = self._function_generate_rational_factor(_varX, _varY, _a, _b)
        else:
            _num, _den, _limit = self._function_generate_rational_boundary(_varX, _varY, _a, _b)
        return sympy.expand(_num*_pre_num), sympy.expand(_den*_pre_den), _limit            
    def _function_generate_polynomial(self, _varX, _varY, _a, _b):
        _poly = 0
        _num_terms = random.randint(self.term_min, self.term_max)
        _exps = flatten_list([[[_i,_j] for _j in range(self.deg_min, self.deg_max+1)] for _i in range(self.deg_min, self.deg_max+1)])
        _terms = random.sample(_exps,k=_num_terms)        
        for _es in _terms:
            _poly += nonzero_randint(self.elem_min, self.elem_max)*_varX**_es[0]*_varY**_es[1]
        return _poly, 1, _poly.subs(_varX,_a).subs(_varY,_b)        
    def _function_generate_rational_domain(self, _varX, _varY, _a, _b):
        _num = 0
        _num_terms = random.randint(self.term_min, self.term_max)
        _exps = flatten_list([[[_i,_j] for _j in range(self.deg_min, self.deg_max+1)] for _i in range(self.deg_min, self.deg_max+1)])
        _terms = random.sample(_exps,k=_num_terms)        
        for _es in _terms:
            _num += nonzero_randint(self.elem_min, self.elem_max)*_varX**_es[0]*_varY**_es[1]
        _den = 0
        _num_terms = random.randint(self.term_min, self.term_max)
        _exps = flatten_list([[[_i,_j] for _j in range(self.deg_min, self.deg_max+1)] for _i in range(self.deg_min, self.deg_max+1)])
        _terms = random.sample(_exps,k=_num_terms)        
        for _es in _terms:
            _den += nonzero_randint(self.elem_min, self.elem_max)*_varX**_es[0]*_varY**_es[1]
        if sympy.degree(_den, _varX) + sympy.degree(_den, _varY) <= 0:
            return self._function_generate_rational_domain(_varX, _varY, _a, _b)
        elif sympy.degree(_num, _varX) + sympy.degree(_num, _varY) <= 0:
            return self._function_generate_rational_domain(_varX, _varY, _a, _b)
        elif _den.subs(_varX,_a).subs(_varY,_b) == 0:
            return self._function_generate_rational_domain(_varX, _varY, _a, _b)
        else:
            return _num, _den, _num.subs(_varX,_a).subs(_varY,_b)/_den.subs(_varX,_a).subs(_varY,_b)
    def _function_generate_rational_factor(self, _varX, _varY, _a, _b):
        _num = 0
        _num_terms = random.randint(self.term_min, self.term_max)
        _exps = flatten_list([[[_i,_j] for _j in range(self.deg_min, self.deg_max+1)] for _i in range(self.deg_min, self.deg_max+1)])
        _terms = random.sample(_exps,k=_num_terms)        
        for _es in _terms:
            _num += nonzero_randint(self.elem_min, self.elem_max)*_varX**_es[0]*_varY**_es[1]
        _den_nonzero = 0
        _num_terms = random.randint(self.term_min, self.term_max)
        _exps = flatten_list([[[_i,_j] for _j in range(self.deg_min, self.deg_max+1)] for _i in range(self.deg_min, self.deg_max+1)])
        _terms = random.sample(_exps,k=_num_terms)  
        for _es in _terms:
            _den_nonzero += nonzero_randint(self.elem_min, self.elem_max)*_varX**_es[0]*_varY**_es[1]
        if _a == 0 and _b == 0:
            _den_zero = (_varX - _a)*random.choice([1,-1]) + (_varY - _b)*random.choice([1,-1])
        elif _a != 0:
            _den_zero = (_varX - _a)*random.choice([1,-1])
        elif _b != 0:
            _den_zero = (_varY - _b)*random.choice([1,-1])
        if sympy.degree(_den_nonzero, _varX) + sympy.degree(_den_nonzero, _varY) <= 0:
            return self._function_generate_rational_factor(_varX, _varY, _a, _b)
        elif _den_nonzero.subs(_varX,_a).subs(_varY,_b) == 0:
            return self._function_generate_rational_factor(_varX, _varY, _a, _b)
        else:
            return sympy.expand(_num*_den_zero), sympy.expand(_den_nonzero*_den_zero), (_num/_den_nonzero).subs(_varX,_a).subs(_varY,_b)
    def _function_generate_rational_boundary(self, _varX, _varY, _a, _b, _is_limit_exists=None):
        if _is_limit_exists is None:
            _is_limit_exists = True if random.random() <= self.limit_exist_ratio else False
        _deg_den = random.randint(self.deg_min, self.deg_max)
        if _is_limit_exists:
            _deg_num = _deg_den + 1
        else:
            _deg_num = _deg_den
        _num = 0
        _num_terms = random.randint(self.term_min, self.term_max)
        _exps = [[_i,_deg_num-_i] for _i in range(_deg_num+1)]
        if _num_terms < len(_exps):
            _terms = random.sample(_exps,k=_num_terms)
        else:
            _terms = _exps
        for _es in _terms:
            _num += nonzero_randint(self.elem_min, self.elem_max)*(_varX-_a)**_es[0]*(_varY-_b)**_es[1]
        _den = 0
        _num_terms = random.randint(self.term_min, self.term_max)
        _exps = [[_i,_deg_den-_i] for _i in range(_deg_den+1)]
        if _num_terms < len(_exps):
            _terms = random.sample(_exps,k=_num_terms)
        else:
            _terms = _exps
        for _es in _terms:
            _den += nonzero_randint(self.elem_min, self.elem_max)*(_varX-_a)**_es[0]*(_varY-_b)**_es[1]
        if sympy.degree(_num, _varX)*sympy.degree(_num, _varY)*sympy.degree(_den, _varX)*sympy.degree(_den, _varY) == 0:
            return self._function_generate_rational_boundary(_varX, _varY, _a, _b, _is_limit_exists)
        if _is_limit_exists:
            return _num, _den, 0
        else:
            _varM = sympy.Symbol('m')
            _limit = sympy.expand(sympy.cancel((_num/_den).subs(_varX,_varM).subs(_varY,_varM)))
            if _limit.is_polynomial(_varM):
                if sympy.degree(_limit, _varM) <= 0:
                    return self._function_generate_rational_boundary(_varX, _varY, _a, _b, _is_limit_exists)
            return _num, _den, r'極限は存在しない'
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='2変数有理関数の極限（sinやlogに帰着）', quiz_number=_quiz_number)
        _varX = sympy.Symbol('x')
        _varY = sympy.Symbol('y')
        _a = random.choice(list(range(self.a_min, self.a_max+1)))
        _b = random.choice(list(range(self.a_min, self.a_max+1)))
        _num, _den, _limit = self._function_generate(_varX, _varY, _a, _b)
        quiz.quiz_identifier = hash(str(_num) + str(_den) + str(_a) + str(_b) + str(_limit))
        # 正答の選択肢の生成
        quiz.data = [_varX, _varY, _num, _den, _a, _b, _limit]
        ans = { 'fraction': 100, 'data': _limit }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_varX, _varY, _num, _den, _a, _b, _limit] = quiz.data
        if isinstance(_limit, str):
            ans['feedback'] = r'因数分解などで分離した上で，\( y=mx \)などの変換を行い，本当に，近づけ方によらずに一定の値に近づくのかを確認しましょう。'
            ans['data'] = 0
            answers.append(dict(ans))
            ans['data'] = sympy.S.Infinity
            answers.append(dict(ans))
            ans['data'] = -sympy.S.Infinity
            answers.append(dict(ans))
            ans['data'] = 1
            answers.append(dict(ans))
            ans['data'] = -1
            answers.append(dict(ans))
        else:
            ans['feedback'] = r'因数分解などで分離した上で，代入操作や式の形によっては極座標変換などで挟み込みなどを検討してみましょう。'
            ans['data'] = r'極限は存在しない'
            answers.append(dict(ans))
            ans['data'] = sympy.S.Infinity
            answers.append(dict(ans))
            ans['data'] = -sympy.S.Infinity
            answers.append(dict(ans))
            ans['data'] = 1
            answers.append(dict(ans))
            ans['data'] = -1
            answers.append(dict(ans))
            if _limit != 0:
                ans['data'] = -_limit
                answers.append(dict(ans))
                ans['data'] = 1/_limit
                answers.append(dict(ans))
                ans['data'] = -1/_limit
                answers.append(dict(ans))
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_varX, _varY, _num, _den, _a, _b, _limit] = quiz.data
        _text = r'次の極限を選択してください。<br />'
        _text += r'\( \lim_{(x,y)\rightarrow(' + sympy.latex(_a) + r',' + sympy.latex(_b) + r')}\frac{'
        _text += sympy.latex(_num, order='lex') + r'}{' + sympy.latex(_den, order='lex') + r'} \)'
        return _text
    def answer_text(self, ans):
        if isinstance(ans['data'], str):
            return ans['data']
        else:
            return r'\( ' +  sympy.latex(ans['data'], order='lex') + r' \)'


# In[118]:


if __name__ == "__main__":
    q = limit_with_sin_or_log_over_xpy()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=25)


# In[119]:


if __name__ == "__main__":
    pass
    #qz.save('limit_with_sin_or_log_over_xpy.xml')


# ## PartialDifferentiableFunction class for the following quizzes

# In[196]:


class PartialDifferentiableFunction():
    _constA = sympy.Symbol('A') # non-zero integer
    _constB = sympy.Symbol('B') # non-zero integer
    _n = sympy.Symbol('n') # positive integer
    _m = sympy.Symbol('m') # positive integer > 1
    _k = sympy.Symbol('k') # positive integer < _m
    _x = sympy.Symbol('x')
    _y = sympy.Symbol('y')
    _z = sympy.Symbol('z')
    # format: [f, [df/dx, df/dy], [incorrect dfs by x, incorrect dfs by y]] 
    _term_types = ['x', 'y', 'xy']
    _term_defs = dict()
    _term_defs['x'] = [_constA*_x**_n, [_n*_constA*_x**(_n-1),0], [[_constA*_x**_n,_n*_constA*_x**_n,0], [_n*_constA*_x**(_n-1)]]]
    _term_defs['y'] = [_constA*_y**_n, [0,_n*_constA*_y**(_n-1)], [[_n*_constA*_y**(_n-1)], [_constA*_y**_n, _n*_constA*_y**_n,0]]]
    _term_defs['xy'] = [_constA*_x**_m*_y**_k, [_m*_constA*_x**(_m-1)*_y**_k,_k*_constA*_x**_m*_y**(_k-1)], [[_k*_constA*_x**_m*_y**(_k-1),_m*_k*_constA*_x**(_m-1)*_y**(_k-1)],[_m*_constA*_x**(_m-1)*_y**_k,_m*_k*_constA*_x**(_m-1)*_y**(_k-1)]]]
    # format: [f, df, incorrect dfs] 
    _composition_types = ['monomial', 'polynomial', 'rational', 'sine', 'cosine', 'natural_exponent']
    _composition_defs = dict()
    _composition_defs['constant'] = [_constA, 0, [_constA, 1]]
    _composition_defs['monomial'] = [_z, 1, [_z, 0]]
    _composition_defs['polynomial'] = [_z**2+_constA*_z+_constB, 2*_z+_constA, [_z+_constA,_z,2*_z**2+_constA*_z]]
    _composition_defs['rational'] = [(_z+_constA)/(_constB*_z), -_constA/(_constB*_z**2), [_constA/(_constB*_z**2), _constA/(_constB*_z)]]
    _composition_defs['sine'] = [sympy.sin(_z), sympy.cos(_z), [sympy.sin(_z), -sympy.cos(_z)]]
    _composition_defs['cosine'] = [sympy.cos(_z), -sympy.sin(_z), [sympy.cos(_z), sympy.sin(_z)]]
    _composition_defs['natural_exponent'] = [sympy.exp(_z), sympy.exp(_z), [sympy.exp(_z-1)]]
    # func = scalar*composition
    # composition = elemental_func @@ terms
    def __init__(self, emin=-3, emax=3, nmin=1, nmax=2, tmin=1, tmax=2, mrate=0.3):
        # 生成する個々の係数等の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 生成する次数の範囲
        self.deg_min = nmin
        self.deg_max = nmax
        # 合成前の多項式に表れる単項式の個数
        self.term_min = tmin
        self.term_max = tmax
        # それぞれの積の微分法の確率
        self.multiplication_ratio = mrate
        # internals
        self.function_internal = 0
        self.function = 0
    # func = scalar*composition | scalar*multiplication
    # multiplication = composition*composition
    # composition = elemental_func @@ terms
    _expression_types = ['summation', 'scalar', 'composition', 'multiplication']
    _incorrect_reasons = ['formula', 'sign', 'composition', 'scalar']
    def generate_function(self):
        _scalar = nonzero_randint(self.elem_min, self.elem_max)
        if random.random() <= self.multiplication_ratio:
            _types = ['monomial', random.choice(self._composition_types[1:])]
            _func = ['scalar', _scalar, self._generate_multiplication(_types)]
        else:
            _func = ['scalar', _scalar, self._generate_composition()]
        self.function_internal = _func
        self.function = self._get_function(self.function_internal)
        if self.function.is_real:
            self.generate_function()
    # multiplication = composition*composition
    def _generate_multiplication(self, _types):
        _func1 = self._generate_composition(_types[0])
        _func2 = self._generate_composition(_types[1])
        return ['multiplication', _func1, _func2]
    # composition = elemental_func @@ terms
    def _generate_composition(self, _type=None):
        if _type is None:
            _type = random.choice(self._composition_types)
        _cfunc = self._generate_elemental_func(_type)
        _tfunc = self._generate_terms()
        return ['composition', _cfunc, _tfunc]
    def _mysubs(self, target, var, repl):
        if sympy.sympify(target).is_real:
            return target
        elif len(var) == 1:
            return target.subs(var[0], repl[0])
        else:
            return self._mysubs(target.subs(var[0], repl[0]), var[1:], repl[1:])
    def _generate_elemental_func(self, _type):
        _vars = [self._constA, self._constB, self._n, self._m, self._k]
        _m = random.randint(max(2,self.deg_min), max(2,self.deg_max))
        _k = random.randint(1, _m-1)
        _repl = [nonzero_randint(self.elem_min, self.elem_max), nonzero_randint(self.elem_min, self.elem_max), 
                 abs(nonzero_randint(self.deg_min, self.deg_max)), _m-_k, _k]
        _cfunc = self._composition_defs[_type]
        _cfunc = [self._mysubs(_cfunc[0], _vars, _repl), 
                  self._mysubs(_cfunc[1], _vars, _repl), 
                  [self._mysubs(_f, _vars, _repl) for _f in _cfunc[2]]]
        return _cfunc
    def _generate_terms(self):
        _nt = random.randint(self.term_min, self.term_max)
        _tfunc = ['summation']
        for _i in range(_nt):
            _vars = [self._constA, self._constB, self._n, self._m, self._k]
            _m = random.randint(max(2,self.deg_min), max(2,self.deg_max))
            _k = random.randint(1, _m-1)
            _repl = [nonzero_randint(self.elem_min, self.elem_max), nonzero_randint(self.elem_min, self.elem_max), 
                     abs(nonzero_randint(self.deg_min, self.deg_max)), _m-_k, _k]
            _type = random.choice(self._term_types)
            _mfunc = self._term_defs[_type]
            _mfunc = [self._mysubs(_mfunc[0], _vars, _repl), 
                      [self._mysubs(_f, _vars, _repl) for _f in _mfunc[1]],
                      [[self._mysubs(_f, _vars, _repl) for _f in _fs] for _fs in _mfunc[2]]]
            _tfunc.append(_mfunc)
        return _tfunc
    def _get_function(self, _func):
        _recursive_call = self._get_function
        # ['summation', 'scalar', 'composition', 'multiplication']
        if _func[0] == 'summation':
            _summand = 0
            for _f in _func[1:]:
                _summand = _summand + _recursive_call(_f)
            return _summand
        elif _func[0] == 'scalar':
            return _func[1] * _recursive_call(_func[2])
        elif _func[0] == 'composition':
            return _recursive_call(_func[1]).subs(self._z, _recursive_call(_func[2]))
        elif _func[0] == 'multiplication':
            return _recursive_call(_func[1]) * _recursive_call(_func[2])
        else:
            return _func[0]
    def get_function(self):
        if self.function_internal == 0:
            self.generate_function()
        return self.function    
    def _get_derivative(self, _func):
        _recursive_call = self._get_derivative
        # ['summation', 'scalar', 'composition', 'multiplication']
        if _func[0] == 'summation':
            _summand = [0,0]
            for _f in _func[1:]:
                _rest = _recursive_call(_f)
                for _i in range(2):
                    _summand[_i] = _summand[_i] + _rest[_i]
            return _summand
        elif _func[0] == 'scalar':
            return [_func[1] * _f for _f in _recursive_call(_func[2])]
        elif _func[0] == 'composition':
            _dz = self._mysubs(_func[1][1], [self._z], [self._get_function(_func[2])])
            return [_f*_dz for _f in _recursive_call(_func[2])]
        elif _func[0] == 'multiplication':
            _fl = [_f * self._get_function(_func[2]) for _f in _recursive_call(_func[1])]
            _fr = [self._get_function(_func[1]) * _f for _f in _recursive_call(_func[2])]
            return [_fl[0] + _fr[0], _fl[1] + _fr[1]]
        else:
            return _func[1]
    def get_derivative(self):
        if self.function == 0:
            self.generate_function()
        return self._get_derivative(self.function_internal)
    def get_higher_derivative(self, dvars):
        if self.function == 0:
            self.generate_function()
        if len(dvars) <= 0:
            return self.get_function()
        else:
            _df = self.get_derivative()
            if dvars[0] == 'x':
                _df = _df[0]
            else:
                _df = _df[1]
            for _c in dvars[1:]:
                if _c == 'x':
                    _df = sympy.diff(_df, self._x)
                else:
                    _df = sympy.diff(_df, self._y)
            return _df
    def _get_incorrect_Xderivatives_by_formula(self, _func):
        _recursive_call = self._get_incorrect_Xderivatives_by_formula
        # ['summation', 'scalar', 'composition', 'multiplication']
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
        elif _func[0] == 'composition':
            _dz = self._mysubs(_func[1][1], [self._z], [self._get_function(_func[2])])
            return [_f*_dz for _f in _recursive_call(_func[2])]
        elif _func[0] == 'multiplication':
            _fls = [_f * self._get_function(_func[2]) for _f in _recursive_call(_func[1])]
            _frs = [self._get_function(_func[1]) * _f for _f in _recursive_call(_func[2])]
            _funcs = []
            for _fl in _fls:
                for _fr in _frs:
                    _funcs.append(_fl+_fr)
            return _funcs
        else:
            return _func[2][0]
    def _get_incorrect_Yderivatives_by_formula(self, _func):
        _recursive_call = self._get_incorrect_Yderivatives_by_formula
        # ['summation', 'scalar', 'composition', 'multiplication']
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
        elif _func[0] == 'composition':
            _dz = self._mysubs(_func[1][1], [self._z], [self._get_function(_func[2])])
            return [_f*_dz for _f in _recursive_call(_func[2])]
        elif _func[0] == 'multiplication':
            _fls = [_f * self._get_function(_func[2]) for _f in _recursive_call(_func[1])]
            _frs = [self._get_function(_func[1]) * _f for _f in _recursive_call(_func[2])]
            _funcs = []
            for _fl in _fls:
                for _fr in _frs:
                    _funcs.append(_fl+_fr)
            return _funcs
        else:
            return _func[2][1]
    def _get_incorrect_Xderivatives_by_sign(self, _func):
        _recursive_call = self._get_incorrect_Xderivatives_by_sign
        # ['summation', 'scalar', 'composition', 'multiplication']
        if _func[0] == 'summation':
            _summand = [0]
            for _f in _func[1:]:
                _dfs = _recursive_call(_f)
                _new_summand = []
                for _df in _dfs:
                    _new_summand = _new_summand + [ _s + random.choice([-1,1])*_df for _s in _summand]
                _summand = _new_summand
            return _summand
        elif _func[0] == 'scalar':
            return [_func[1] * _f for _f in _recursive_call(_func[2])]
        elif _func[0] == 'composition':
            _dz = self._mysubs(_func[1][1], [self._z], [self._get_function(_func[2])])
            return [_f*_dz for _f in _recursive_call(_func[2])]
        elif _func[0] == 'multiplication':
            _fls = [_f * self._get_function(_func[2]) for _f in _recursive_call(_func[1])]
            _frs = [self._get_function(_func[1]) * _f for _f in _recursive_call(_func[2])]
            _funcs = []
            for _fl in _fls:
                for _fr in _frs:
                    _funcs.append(_fl-_fr)
            return _funcs
        else:
            return [_func[1][0]]
    def _get_incorrect_Yderivatives_by_sign(self, _func):
        _recursive_call = self._get_incorrect_Yderivatives_by_sign
        # ['summation', 'scalar', 'composition', 'multiplication']
        if _func[0] == 'summation':
            _summand = [0]
            for _f in _func[1:]:
                _dfs = _recursive_call(_f)
                _new_summand = []
                for _df in _dfs:
                    _new_summand = _new_summand + [ _s + random.choice([-1,1])*_df for _s in _summand]
                _summand = _new_summand
            return _summand
        elif _func[0] == 'scalar':
            return [_func[1] * _f for _f in _recursive_call(_func[2])]
        elif _func[0] == 'composition':
            _dz = self._mysubs(_func[1][1], [self._z], [self._get_function(_func[2])])
            return [_f*_dz for _f in _recursive_call(_func[2])]
        elif _func[0] == 'multiplication':
            _fls = [_f * self._get_function(_func[2]) for _f in _recursive_call(_func[1])]
            _frs = [self._get_function(_func[1]) * _f for _f in _recursive_call(_func[2])]
            _funcs = []
            for _fl in _fls:
                for _fr in _frs:
                    _funcs.append(_fl-_fr)
            return _funcs
        else:
            return [_func[1][1]]
    def _get_incorrect_Xderivatives_by_composition(self, _func):
        _recursive_call = self._get_incorrect_Xderivatives_by_composition
        # ['summation', 'scalar', 'composition', 'multiplication']
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
        elif _func[0] == 'composition':
            _dz = self._mysubs(_func[1][1], [self._z], [self._get_function(_func[2])])
            return _recursive_call(_func[2]) + [_dz]
        elif _func[0] == 'multiplication':
            _fls = [_f * self._get_function(_func[2]) for _f in _recursive_call(_func[1])]
            _frs = [self._get_function(_func[1]) * _f for _f in _recursive_call(_func[2])]
            _funcs = []
            for _fl in _fls:
                for _fr in _frs:
                    _funcs.append(_fl+_fr)
            return _funcs
        else:
            return [_func[1][0]]
    def _get_incorrect_Yderivatives_by_composition(self, _func):
        _recursive_call = self._get_incorrect_Yderivatives_by_composition
        # ['summation', 'scalar', 'composition', 'multiplication']
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
        elif _func[0] == 'composition':
            _dz = self._mysubs(_func[1][1], [self._z], [self._get_function(_func[2])])
            return _recursive_call(_func[2]) + [_dz]
        elif _func[0] == 'multiplication':
            _fls = [_f * self._get_function(_func[2]) for _f in _recursive_call(_func[1])]
            _frs = [self._get_function(_func[1]) * _f for _f in _recursive_call(_func[2])]
            _funcs = []
            for _fl in _fls:
                for _fr in _frs:
                    _funcs.append(_fl+_fr)
            return _funcs
        else:
            return [_func[1][1]]
    def _get_incorrect_Xderivatives_by_scalar(self, _func):
        _recursive_call = self._get_incorrect_Xderivatives_by_scalar
        # ['summation', 'scalar', 'composition', 'multiplication']
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
        elif _func[0] == 'composition':
            _dz = self._mysubs(_func[1][1], [self._z], [self._get_function(_func[2])])
            return [_f*_dz for _f in _recursive_call(_func[2])]
        elif _func[0] == 'multiplication':
            _fls = [_f * self._get_function(_func[2]) for _f in _recursive_call(_func[1])]
            _frs = [self._get_function(_func[1]) * _f for _f in _recursive_call(_func[2])]
            _funcs = []
            for _fl in _fls:
                for _fr in _frs:
                    _funcs.append(_fl+_fr)
            return _funcs
        else:
            return [_func[1][0]]
    def _get_incorrect_Yderivatives_by_scalar(self, _func):
        _recursive_call = self._get_incorrect_Yderivatives_by_scalar
        # ['summation', 'scalar', 'composition', 'multiplication']
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
        elif _func[0] == 'composition':
            _dz = self._mysubs(_func[1][1], [self._z], [self._get_function(_func[2])])
            return [_f*_dz for _f in _recursive_call(_func[2])]
        elif _func[0] == 'multiplication':
            _fls = [_f * self._get_function(_func[2]) for _f in _recursive_call(_func[1])]
            _frs = [self._get_function(_func[1]) * _f for _f in _recursive_call(_func[2])]
            _funcs = []
            for _fl in _fls:
                for _fr in _frs:
                    _funcs.append(_fl+_fr)
            return _funcs
        else:
            return [_func[1][1]]
    def _not_same_check(self, value):
        if not value.is_real:
            return False
        if value > 1.0e-4:
            return True
        return False
    def _incorrect_Xderivarives_only(self, _dfs):
        _funcs = []
        _correct_df = self.get_derivative()[0]
        _correct_df0 = self._mysubs(_correct_df, [self._x, self._y], [0, 0])
        _correct_df1 = self._mysubs(_correct_df, [self._x, self._y], [-1, 1])
        _correct_df2 = self._mysubs(_correct_df, [self._x, self._y], [1, -1])
        for _df in _dfs:
            if _df in _funcs:
                continue
            if self._not_same_check(abs(sympy.N(_correct_df0 - self._mysubs(_df, [self._x, self._y], [0, 0])))):
                _funcs.append(_df)
            elif self._not_same_check(abs(sympy.N(_correct_df1 - self._mysubs(_df, [self._x, self._y], [-1, 1])))):
                _funcs.append(_df)
            elif self._not_same_check(abs(sympy.N(_correct_df2 - self._mysubs(_df, [self._x, self._y], [1, -1])))):
                _funcs.append(_df)
        return _funcs
    def _incorrect_Yderivarives_only(self, _dfs):
        _funcs = []
        _correct_df = self.get_derivative()[1]
        _correct_df0 = self._mysubs(_correct_df, [self._x, self._y], [0, 0])
        _correct_df1 = self._mysubs(_correct_df, [self._x, self._y], [-1, 1])
        _correct_df2 = self._mysubs(_correct_df, [self._x, self._y], [1, -1])
        for _df in _dfs:
            if _df in _funcs:
                continue
            if self._not_same_check(abs(sympy.N(_correct_df0 - self._mysubs(_df, [self._x, self._y], [0, 0])))):
                _funcs.append(_df)
            elif self._not_same_check(abs(sympy.N(_correct_df1 - self._mysubs(_df, [self._x, self._y], [-1, 1])))):
                _funcs.append(_df)
            elif self._not_same_check(abs(sympy.N(_correct_df2 - self._mysubs(_df, [self._x, self._y], [1, -1])))):
                _funcs.append(_df)
        return _funcs
    def get_incorrect_Xderivatives(self, _type=None):
        if self.function == 0:
            self.generate_function()
        if _type is None:
            _type = random.choice(self._incorrect_reasons)
        if _type == 'formula':
            _dfs = self._get_incorrect_Xderivatives_by_formula(self.function_internal)
        elif _type == 'sign':
            _dfs = self._get_incorrect_Xderivatives_by_sign(self.function_internal)
        elif _type == 'composition':
            _dfs = self._get_incorrect_Xderivatives_by_composition(self.function_internal)
        elif _type == 'scalar':
            _dfs = self._get_incorrect_Xderivatives_by_scalar(self.function_internal)
        return self._incorrect_Xderivarives_only(_dfs)
    def get_incorrect_Yderivatives(self, _type=None):
        if self.function == 0:
            self.generate_function()
        if _type is None:
            _type = random.choice(self._incorrect_reasons)
        if _type == 'formula':
            _dfs = self._get_incorrect_Yderivatives_by_formula(self.function_internal)
        elif _type == 'sign':
            _dfs = self._get_incorrect_Yderivatives_by_sign(self.function_internal)
        elif _type == 'composition':
            _dfs = self._get_incorrect_Yderivatives_by_composition(self.function_internal)
        elif _type == 'scalar':
            _dfs = self._get_incorrect_Yderivatives_by_scalar(self.function_internal)
        return self._incorrect_Yderivarives_only(_dfs)
    def _incorrect_higher_derivarives_only(self, _dfs, dvars):
        _funcs = []
        _correct_df = self.get_higher_derivative(dvars)
        _correct_df0 = self._mysubs(_correct_df, [self._x, self._y], [0, 0])
        _correct_df1 = self._mysubs(_correct_df, [self._x, self._y], [-1, 1])
        _correct_df2 = self._mysubs(_correct_df, [self._x, self._y], [1, -1])
        for _df in _dfs:
            if _df in _funcs:
                continue
            if self._not_same_check(abs(sympy.N(_correct_df0 - self._mysubs(_df, [self._x, self._y], [0, 0])))):
                _funcs.append(_df)
            elif self._not_same_check(abs(sympy.N(_correct_df1 - self._mysubs(_df, [self._x, self._y], [-1, 1])))):
                _funcs.append(_df)
            elif self._not_same_check(abs(sympy.N(_correct_df2 - self._mysubs(_df, [self._x, self._y], [1, -1])))):
                _funcs.append(_df)
        return _funcs
    def get_incorrect_higher_derivatives_by_dvars(self, dvars):
        _dfs = []
        _nx = dvars.count(r'x')
        _ny = dvars.count(r'y')
        for _ix in range(0,len(dvars)+1):
            for _iy in range(0,len(dvars)+1):
                if _ix + _iy > len(dvars):
                    continue
                _dvars = r''
                for _i in range(_ix):
                    _dvars += r'x'
                for _i in range(_iy):
                    _dvars += r'y'
                _dfs.append(self.get_higher_derivative(_dvars))
        return _dfs        
    def get_incorrect_higher_derivatives(self, dvars, _type=None):
        if self.function == 0:
            self.generate_function()
        if _type is None:
            _type = random.choice(self._incorrect_reasons)
        if _type == 'dvars':
            _dfs = self.get_incorrect_higher_derivatives_by_dvars(dvars)
        elif _type == 'formula':
            _dfs = self._get_incorrect_Xderivatives_by_formula(self.function_internal)
            _dfs = _dfs + self._get_incorrect_Yderivatives_by_formula(self.function_internal)
        elif _type == 'sign':
            _dfs = self._get_incorrect_Xderivatives_by_sign(self.function_internal)
            _dfs = _dfs + self._get_incorrect_Yderivatives_by_sign(self.function_internal)
        elif _type == 'composition':
            _dfs = self._get_incorrect_Xderivatives_by_composition(self.function_internal)
            _dfs = _dfs + self._get_incorrect_Yderivatives_by_composition(self.function_internal)
        elif _type == 'scalar':
            _dfs = self._get_incorrect_Xderivatives_by_scalar(self.function_internal)
            _dfs = _dfs + self._get_incorrect_Yderivatives_by_scalar(self.function_internal)
        return self._incorrect_higher_derivarives_only(_dfs, dvars)


# In[168]:


if __name__ == "__main__":
    df = PartialDifferentiableFunction()
    df.generate_function()
    display(df.function_internal)
    display(df.get_function())
    _dfs = df.get_derivative()
    display(_dfs[0])
    display(_dfs[1])
    display(df.get_incorrect_Xderivatives())
    display(df.get_incorrect_Yderivatives())
    display(df.get_higher_derivative("xy"))
    display(df.get_incorrect_higher_derivatives("xy"))


# ## partial derivative

# In[183]:


class partial_derivative(core.Question):
    name = '偏導関数'
    def __init__(self, emin=-3, emax=3, nmin=1, nmax=2, tmin=1, tmax=2, mrate=0.3):
        # 生成する個々の係数等の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 生成する次数の範囲
        self.deg_min = nmin
        self.deg_max = nmax
        # 合成前の多項式に表れる単項式の個数
        self.term_min = tmin
        self.term_max = tmax
        # それぞれの積の微分法の確率
        self.multiplication_ratio = mrate
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='偏導関数', quiz_number=_quiz_number)
        df = PartialDifferentiableFunction(emin=self.elem_min, emax=self.elem_max, nmin=self.deg_min, nmax=self.deg_max, 
                                           tmin=self.term_min, tmax=self.term_max, mrate=self.multiplication_ratio)
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
        # 'formula', 'sign', 'composition', 'scalar'
        ans['feedback'] = r'変数の取り違えや，基本的な関数の導関数の公式を確認してください。'
        for _incorrectX in df.get_incorrect_Xderivatives('formula') + [_df[0]]:
            for _incorrectY in df.get_incorrect_Yderivatives('formula') + [_df[1]]:
                if _df[0] != _incorrectX or _df[1] != _incorrectY:
                    ans['data'] = [_incorrectX, _incorrectY]
                    answers.append(dict(ans))
        ans['feedback'] = r'符号に注意して計算を行ってください。'
        for _incorrectX in df.get_incorrect_Xderivatives('sign') + [_df[0]]:
            for _incorrectY in df.get_incorrect_Yderivatives('sign') + [_df[1]]:
                if _df[0] != _incorrectX or _df[1] != _incorrectY:
                    ans['data'] = [_incorrectX, _incorrectY]
                    answers.append(dict(ans))
        ans['feedback'] = r'係数などのかけ忘れなどに注意してください。'
        for _incorrectX in df.get_incorrect_Xderivatives('scalar') + [_df[0]]:
            for _incorrectY in df.get_incorrect_Yderivatives('scalar') + [_df[1]]:
                if _df[0] != _incorrectX or _df[1] != _incorrectY:
                    ans['data'] = [_incorrectX, _incorrectY]
                    answers.append(dict(ans))
        ans['feedback'] = r'各種の微分法などを確認しましょう。取り違えや取りこぼしがあります。'
        for _incorrectX in df.get_incorrect_Xderivatives('composition') + [_df[0]]:
            for _incorrectY in df.get_incorrect_Yderivatives('composition') + [_df[1]]:
                if _df[0] != _incorrectX or _df[1] != _incorrectY:
                    ans['data'] = [_incorrectX, _incorrectY]
                    answers.append(dict(ans))
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_func, _df, df] = quiz.data
        _text = r'次の関数\( f(x,y) \)の1階の偏導関数を選択してください。'
        _text += r'<br />\( f(x,y)=' + sympy.latex(_func, order='lex') + r' \)'
        return _text
    def answer_text(self, ans):
        _text = r'\( \frac{\partial f}{\partial x}=' +  sympy.latex(ans['data'][0], order='lex')
        _text += r',\;\frac{\partial f}{\partial y}=' +  sympy.latex(ans['data'][1], order='lex') + r' \)'
        return _text


# In[184]:


if __name__ == "__main__":
    q = partial_derivative(emin=-3, emax=3, nmin=2, nmax=3, tmin=1, tmax=1, mrate=0.0)
    q.name = r'偏導関数（比較的簡単なもの）'
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=25)


# In[185]:


if __name__ == "__main__":
    pass
    #qz.save('partial_derivative.xml')


# In[187]:


if __name__ == "__main__":
    q = partial_derivative(emin=-2, emax=2, nmin=2, nmax=2, tmin=1, tmax=1, mrate=1.0)
    q.name = r'偏導関数（比較的複雑なもの）'
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=25)


# In[188]:


if __name__ == "__main__":
    pass
    #qz.save('partial_derivative_2.xml')


# ## partial higher derivative

# In[198]:


class partial_higher_derivative(core.Question):
    name = '高階偏導関数'
    def __init__(self, emin=-3, emax=3, nmin=1, nmax=2, tmin=1, tmax=2, mrate=0.3, hmin=2, hmax=3):
        # 生成する個々の係数等の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 生成する次数の範囲
        self.deg_min = nmin
        self.deg_max = nmax
        # 合成前の多項式に表れる単項式の個数
        self.term_min = tmin
        self.term_max = tmax
        # それぞれの積の微分法の確率
        self.multiplication_ratio = mrate
        # 何階の高階偏導関数を求めるか
        self.nth_min = hmin
        self.nth_max = hmax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='高階偏導関数', quiz_number=_quiz_number)
        df = PartialDifferentiableFunction(emin=self.elem_min, emax=self.elem_max, nmin=self.deg_min, nmax=self.deg_max, 
                                           tmin=self.term_min, tmax=self.term_max, mrate=self.multiplication_ratio)
        _func = df.get_function()
        _n = random.randint(self.nth_min, self.nth_max)
        _dvars = r''
        for _i in range(_n):
            _dvars += random.choice([r'x', r'y'])
        quiz.quiz_identifier = hash(str(_func) + str(_dvars))
        # 正答の選択肢の生成
        _df = df.get_higher_derivative(_dvars)
        quiz.data = [_func, _df, df, _dvars]
        ans = { 'fraction': 100, 'data': _df }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_func, _df, df, _dvars] = quiz.data
        ans['feedback'] = r'高階の偏微分なので，指定された変数で順次偏微分してください。'
        for _incorrect in df.get_incorrect_higher_derivatives(_dvars, 'dvars'):
                ans['data'] = _incorrect
                answers.append(dict(ans))
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_func, _df, df, _dvars] = quiz.data
        _text = r'次の関数\( f(x,y) \)の高階偏導関数\( f_{' + _dvars + r'} \)を選択してください。'
        _text += r'<br />\( f(x,y)=' + sympy.latex(_func, order='lex') + r' \)'
        return _text
    def answer_text(self, ans):
        return r'\( ' +  sympy.latex(ans['data'], order='lex') + r' \)'


# In[199]:


if __name__ == "__main__":
    q = partial_higher_derivative(emin=-3, emax=3, nmin=2, nmax=3, tmin=1, tmax=1, mrate=0.0, hmin=2, hmax=2)
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=25)


# In[200]:


if __name__ == "__main__":
    pass
    #qz.save('partial_higher_derivative.xml')


# ## extremum condition

# In[223]:


class extremum_condition(core.Question):
    name = '極値を取るための必要十分条件'
    def __init__(self, emin=-10, emax=10):
        # 生成する個々の係数等の範囲
        self.elem_min = emin
        self.elem_max = emax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='極値の必要十分条件', quiz_number=_quiz_number)
        _a = random.randint(self.elem_min, self.elem_max)
        _b = random.randint(self.elem_min, self.elem_max)
        _fx = random.choice([0,0,0,0,0,1])*random.randint(self.elem_min, self.elem_max)
        _fy = random.choice([0,0,0,0,0,1])*random.randint(self.elem_min, self.elem_max)
        _fxx = nonzero_randint(self.elem_min, self.elem_max)
        _fyy = random.randint(self.elem_min, self.elem_max)
        _fxy = random.randint(sympy.floor(self.elem_min/5), sympy.ceiling(self.elem_max/5))
        _dxy = _fxx*_fyy-_fxy**2
        quiz.quiz_identifier = hash(str(_a) + str(_b) + str(_fx) + str(_fy) + str(_fxx) + str(_fyy) + str(_fxy) + str(_dxy))
        # 正答の選択肢の生成
        if _dxy < 0 or _fx != 0 or _fy != 0:
            _ans = 'no'
        elif _dxy > 0:
            if _fxx > 0:
                _ans = 'min'
            else:
                _ans = 'max'
        else:
            _ans = 'dontknow'        
        quiz.data = [_a, _b, _fx, _fy, _fxx, _fyy, _fxy, _dxy, _ans]
        ans = { 'fraction': 100, 'data': [_a, _b, _ans]}
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_a, _b, _fx, _fy, _fxx, _fyy, _fxy, _dxy, _ans] = quiz.data
        ans['feedback'] = r'極値を取るための必要条件と十分条件をそれぞれ確認してください。'
        if _ans != 'no':
            ans['data'] = [_a, _b, 'no']
            answers.append(dict(ans))
        if _ans != 'min':
            ans['data'] = [_a, _b, 'min']
            answers.append(dict(ans))
        if _ans != 'max':
            ans['data'] = [_a, _b, 'max']
            answers.append(dict(ans))
        if _ans != 'dontknow':
            ans['data'] = [_a, _b, 'dontknow']
            answers.append(dict(ans))
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_a, _b, _fx, _fy, _fxx, _fyy, _fxy, _dxy, _ans] = quiz.data
        _text = r'ある\( C^2 \)級の関数\( f(x,y) \)は，点\( (a,b)=(' + sympy.latex(_a) + r',' + sympy.latex(_b) + r') \)において，'
        _text += r'それぞれ次のような条件を満たしています。極値に関して最も適切な選択肢を選んでください。<br />'
        _text += r'\( f_{x}(a,b)=' + sympy.latex(_fx) + r',\;'
        _text += r'f_{y}(a,b)=' + sympy.latex(_fy) + r',\;'
        _text += r'f_{xx}(a,b)=' + sympy.latex(_fxx) + r',\;'
        _text += r'f_{yy}(a,b)=' + sympy.latex(_fyy) + r',\;'
        _text += r'f_{xy}(a,b)=' + sympy.latex(_fxy) + r' \)'
        return _text
    def answer_text(self, ans):
        [_a, _b, _ans] = ans['data']
        _text = r'関数\( f(x,y) \)は点\( (a,b)=(' + sympy.latex(_a) + r',' + sympy.latex(_b) + r') \)で，'
        if _ans == 'no':
            _text += r'極値を取らない。'
        elif _ans == 'max':
            _text += r'極大値を取る。'
        elif _ans == 'min':
            _text += r'極小値を取る。'
        else:
            _text += r'極値を取るかは確定できない。'
        return _text


# In[224]:


if __name__ == "__main__":
    q = extremum_condition()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=25)


# In[225]:


if __name__ == "__main__":
    pass
    #qz.save('extremum_condition.xml')


# ## extremum of polynomial of total degree 2

# In[231]:


class extremum_of_polynomial(core.Question):
    name = '極値（全次数が2次の2変数多項式の場合の極値問題）'
    def __init__(self, emin=-4, emax=4):
        # 生成する個々の係数等の範囲
        self.elem_min = emin
        self.elem_max = emax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='極値（2変数多項式）', quiz_number=_quiz_number)
        _x = sympy.Symbol('x')
        _y = sympy.Symbol('y')
        _t = random.choice([1,2,3,4])
        _c00 = random.randint(self.elem_min, self.elem_max)
        _c10 = random.randint(self.elem_min, self.elem_max)
        _c11 = random.randint(self.elem_min, self.elem_max)
        _c01 = random.randint(self.elem_min, self.elem_max)
        if _t == 1:
            _f = 4*_c00 + 4*_c10*_x - _x**2 + _c11**2*_x**2 + 4*_c01*_y + 4*_c11*_x*_y + 4*_y**2
            _a = 2*_c10 - _c01*_c11
            _b = (-_c01 - 2*_c10*_c11 + _c01*_c11**2)/sympy.Integer(2)
        elif _t == 2:
            _f = 4*_c00 + 4*_c10*_x + _x**2 - _c11**2*_x**2 + 4*_c01*_y + 4*_c11*_x*_y - 4*_y**2
            _a = -2*_c10 - _c01*_c11
            _b = (_c01 - 2*_c10*_c11 - _c01*_c11**2)/sympy.Integer(2)
        elif _t == 3:
            _f = 4*_c00 + 4*_c10*_x + _x**2 + _c11**2*_x**2 + 4*_c01*_y + 4*_c11*_x*_y + 4*_y**2
            _a = -2*_c10 + _c01*_c11
            _b = (-_c01 + 2*_c10*_c11 - _c01*_c11**2)/sympy.Integer(2)
        else:
            _f = 4*_c00 + 4*_c10*_x - _x**2 - _c11**2*_x**2 + 4*_c01*_y + 4*_c11*_x*_y - 4*_y**2
            _a = 2*_c10 + _c01*_c11
            _b = (_c01 + 2*_c10*_c11 + _c01*_c11**2)/sympy.Integer(2)
        quiz.quiz_identifier = hash(str(_a) + str(_b) + str(_f))
        # 正答の選択肢の生成
        _fxx = sympy.diff(_f,_x,_x).subs(_x,_a).subs(_y,_b)
        _fyy = sympy.diff(_f,_y,_y).subs(_x,_a).subs(_y,_b)
        _fxy = sympy.diff(_f,_x,_y).subs(_x,_a).subs(_y,_b)
        _dxy = _fxx*_fyy-_fxy**2        
        if _dxy < 0:
            _ans = 'no'
        elif _dxy > 0:
            if _fxx > 0:
                _ans = 'min'
            else:
                _ans = 'max'
        else:
            _ans = 'dontknow'        
        quiz.data = [_a, _b, _f, _x, _y, _fxx, _fyy, _fxy, _dxy, _ans]
        ans = { 'fraction': 100, 'data': [_a, _b, _ans]}
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_a, _b, _f, _x, _y, _fxx, _fyy, _fxy, _dxy, _ans] = quiz.data
        ans['feedback'] = r'極値を取るための必要条件と十分条件をそれぞれ確認してください。'
        if _ans != 'no':
            ans['data'] = [_a, _b, 'no']
            answers.append(dict(ans))
        if _ans != 'min':
            ans['data'] = [_a, _b, 'min']
            answers.append(dict(ans))
        if _ans != 'max':
            ans['data'] = [_a, _b, 'max']
            answers.append(dict(ans))
        if _ans != 'dontknow':
            ans['data'] = [_a, _b, 'dontknow']
            answers.append(dict(ans))
        _fx = sympy.diff(_f,_x)
        _fy = sympy.diff(_f,_y)
        _ia = random.randint(self.elem_min, self.elem_max)
        _ib = random.randint(self.elem_min, self.elem_max)
        while _fx.subs(_x,_ia).subs(_y,_ib) == 0:
            _ia = random.randint(self.elem_min, self.elem_max)
            _ib = random.randint(self.elem_min, self.elem_max)
        ans['data'] = [_ia, _ib, 'min']
        answers.append(dict(ans))
        ans['data'] = [_ia, _ib, 'max']
        answers.append(dict(ans))
        ans['data'] = [_ia, _ib, 'dontknow']
        answers.append(dict(ans))
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_a, _b, _f, _x, _y, _fxx, _fyy, _fxy, _dxy, _ans] = quiz.data
        _text = r'次の関数\( f(x,y) \)の極値に関して最も適切な選択肢を選んでください。<br />'
        _text += r'\( f(x,y)=' + sympy.latex(_f) + r' \)'
        return _text
    def answer_text(self, ans):
        [_a, _b, _ans] = ans['data']
        _text = r'関数\( f(x,y) \)は点\( (a,b)=(' + sympy.latex(_a) + r',' + sympy.latex(_b) + r') \)で，'
        if _ans == 'no':
            _text += r'極値を取らない。'
        elif _ans == 'max':
            _text += r'極大値を取る。'
        elif _ans == 'min':
            _text += r'極小値を取る。'
        else:
            _text += r'極値を取るかは確定できない。'
        return _text


# In[232]:


if __name__ == "__main__":
    q = extremum_of_polynomial()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=25)


# In[233]:


if __name__ == "__main__":
    pass
    #qz.save('extremum_of_polynomial.xml')


# ## Lagrangian multipliers

# In[248]:


class Lagrangian_multipliers(core.Question):
    name = 'ラグランジュの未定乗数法（解くべき方程式の確認）'
    def __init__(self, emin=-3, emax=3, nmin=0, nmax=3, tmin=1, tmax=3):
        # 生成する個々の係数等の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 生成する個々の次数の範囲
        self.deg_min = nmin
        self.deg_max = nmax
        # 生成する単項式の個数の範囲
        self.term_min = tmin
        self.term_max = tmax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='ラグランジュの未定乗数法', quiz_number=_quiz_number)
        _x = sympy.Symbol('x')
        _y = sympy.Symbol('y')
        _r = sympy.Symbol('lambda')
        while True:
            _f = 0
            _nt = random.randint(self.term_min, self.term_max)
            _deg = []
            for _i in range(_nt+1):
                for _j in range(_nt+1):
                    if _i + _j <= _nt:
                        _deg.append([_i,_j])
            for _ds in random.sample(_deg,k=_nt):
                _f += nonzero_randint(self.elem_min, self.elem_max)*_x**_ds[0]*_y**_ds[1]
            _phi = 0
            _nt = random.randint(self.term_min, self.term_max)
            _deg = []
            for _i in range(_nt+1):
                for _j in range(_nt+1):
                    if _i + _j <= _nt:
                        _deg.append([_i,_j])
            for _ds in random.sample(_deg,k=_nt):
                _phi += nonzero_randint(self.elem_min, self.elem_max)*_x**_ds[0]*_y**_ds[1]
            _fx = sympy.diff(_f,_x)
            _fy = sympy.diff(_f,_y)
            _phix = sympy.diff(_phi,_x)
            _phiy = sympy.diff(_phi,_y)
            if _fx != _fy and _fx != _phix and _fx != _phiy and _fy != _phix and _fy != _phiy and _phix != _phiy and _f != 0 and _phi != 0 and not _phi.is_real:
                break
        quiz.quiz_identifier = hash(str(_f) + str(_phi))
        # 正答の選択肢の生成
        quiz.data = [_f, _x, _y, _phi, _r, _fx, _fy, _phix, _phiy]
        ans = { 'fraction': 100, 'data': [_fx, _fy, _phi, _phix, _phiy]}
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_f, _x, _y, _phi, _r, _fx, _fy, _phix, _phiy] = quiz.data
        ans['feedback'] = r'ラグランジュの未定乗数法に関して，教科書や資料を見直してください。'
        ans['data'] = [_fx, _fy, _phi, _phiy, _phix]
        answers.append(dict(ans))
        ans['data'] = [_fy, _fx, _phi, _phix, _phiy]
        answers.append(dict(ans))
        ans['data'] = [_fx, _fy, _f, _phix, _phiy]
        answers.append(dict(ans))
        ans['data'] = [_fx, _fy, _phix, _phi, _phiy]
        answers.append(dict(ans))
        ans['data'] = [_fx, _fy, _phiy, _phix, _phi]
        answers.append(dict(ans))
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_f, _x, _y, _phi, _r, _fx, _fy, _phix, _phiy] = quiz.data
        _text = r'次の関数\( f(x,y),\;\varphi(x,y) \)に対して，制約条件\( \varphi(x,y)=0 \)のもとで，\( f(x,y) \)が極値を取る点を調べるために解くべき方程式として適切なものを選んでください。<br />'
        _text += r'\( f(x,y)=' + sympy.latex(_f) + r',\;\varphi(x,y)=' + sympy.latex(_phi) + r' \)'
        return _text
    def answer_text(self, ans):
        [_fx, _fy, _phi, _phix, _phiy] = ans['data']
        _text = r'\( \left\{\begin{array}{l}'
        if _fx != 0 or _phix != 0:
            _text += sympy.latex(_fx) + r'-\lambda\times\left(' + sympy.latex(_phix) + r'\right)=0\\' 
        if _fy != 0 or _phiy != 0:
            _text += sympy.latex(_fy) + r'-\lambda\times\left(' + sympy.latex(_phiy) + r'\right)=0\\' 
        if _phi != 0:
            _text += sympy.latex(_phi) + r'=0' 
        _text += r'\end{array}\right. \)'
        return _text


# In[249]:


if __name__ == "__main__":
    q = Lagrangian_multipliers()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=25)


# In[250]:


if __name__ == "__main__":
    pass
    #qz.save('Lagrangian_multipliers.xml')


# ## derivative of implicit function

# In[257]:


class derivative_of_implicit_function(core.Question):
    name = '陰関数の導関数（陰関数定理）'
    def __init__(self, emin=-3, emax=3, nmin=0, nmax=3, tmin=1, tmax=3):
        # 生成する個々の係数等の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 生成する個々の次数の範囲
        self.deg_min = nmin
        self.deg_max = nmax
        # 生成する単項式の個数の範囲
        self.term_min = tmin
        self.term_max = tmax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='陰関数の導関数', quiz_number=_quiz_number)
        _x = sympy.Symbol('x')
        _y = sympy.Symbol('y')
        while True:
            _f = 0
            _nt = random.randint(self.term_min, self.term_max)
            _deg = []
            for _i in range(_nt+1):
                for _j in range(_nt+1):
                    if _i + _j <= _nt:
                        _deg.append([_i,_j])
            for _ds in random.sample(_deg,k=_nt):
                _f += nonzero_randint(self.elem_min, self.elem_max)*_x**_ds[0]*_y**_ds[1]
            _fx = sympy.diff(_f,_x)
            _fy = sympy.diff(_f,_y)
            if _fx != _fy and _f != 0 and _fx != 0 and _fy != 0:
                break
        quiz.quiz_identifier = hash(str(_f))
        # 正答の選択肢の生成
        quiz.data = [_f, _x, _y, _fx, _fy]
        ans = { 'fraction': 100, 'data': -_fx/_fy}
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_f, _x, _y, _fx, _fy] = quiz.data
        if sympy.expand(_fx + _fx*_fy) != 0:
            ans['feedback'] = r'陰関数定理を確認しましょう。'
            ans['data'] = _fx
            answers.append(dict(ans))
        if sympy.expand(_fx + _fy*_fy) != 0:
            ans['feedback'] = r'陰関数定理を確認しましょう。'
            ans['data'] = _fy
            answers.append(dict(ans))
        ans['feedback'] = r'陰関数定理を確認しましょう。符号がおかしい可能性があります。'
        ans['data'] = _fx/_fy
        answers.append(dict(ans))
        ans['feedback'] = r'陰関数定理を確認しましょう。偏導関数の上下を確認しましょう。'
        ans['data'] = -_fy/_fx
        answers.append(dict(ans))
        ans['feedback'] = r'陰関数定理を確認しましょう。符号や偏導関数の上下を確認しましょう。'
        ans['data'] = _fy/_fx
        answers.append(dict(ans))
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_f, _x, _y, _fx, _fy] = quiz.data
        _text = r'次式で定義される陰関数\( f(x,y)=0 \)に関して，その導関数\( \frac{dy}{dx} \)を求めてください。<br />'
        _text += r'\( f(x,y)=' + sympy.latex(_f) + r'=0 \)'
        return _text
    def answer_text(self, ans):
        return r'\( ' + sympy.latex(ans['data'], order='lex') + r' \)'


# In[258]:


if __name__ == "__main__":
    q = derivative_of_implicit_function()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=25)


# In[259]:


if __name__ == "__main__":
    pass
    #qz.save('derivative_of_implicit_function.xml')


# ## second derivative of implicit function

# In[268]:


class second_derivative_of_implicit_function(core.Question):
    name = '陰関数の2階導関数（陰関数定理）'
    def __init__(self, emin=-3, emax=3, nmin=1, nmax=2, tmin=1, tmax=2):
        # 生成する個々の係数等の範囲
        self.elem_min = emin
        self.elem_max = emax
        # 生成する個々の次数の範囲
        self.deg_min = nmin
        self.deg_max = nmax
        # 生成する単項式の個数の範囲
        self.term_min = tmin
        self.term_max = tmax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='陰関数の2階導関数', quiz_number=_quiz_number)
        _x = sympy.Symbol('x')
        _y = sympy.Symbol('y')
        while True:
            _f = 0
            _nt = random.randint(self.term_min, self.term_max)
            _deg = []
            for _i in range(_nt+1):
                for _j in range(_nt+1):
                    if _i + _j <= _nt:
                        _deg.append([_i,_j])
            for _ds in random.sample(_deg,k=_nt):
                _f += nonzero_randint(self.elem_min, self.elem_max)*_x**_ds[0]*_y**_ds[1]
            _fx = sympy.diff(_f,_x)
            _fy = sympy.diff(_f,_y)
            if _fx != _fy and _f != 0 and _fx != 0 and _fy != 0:
                break
        _fxx = sympy.diff(_f,_x,_x)
        _fyy = sympy.diff(_f,_y,_y)
        _fxy = sympy.diff(_f,_x,_y)
        quiz.quiz_identifier = hash(str(_f))
        # 正答の選択肢の生成
        quiz.data = [_f, _x, _y, _fx, _fy, _fxx, _fyy, _fxy]
        _ans = sympy.expand(2*_fx*_fy*_fxy-_fx*_fx*_fyy-_fy*_fy*_fxx)
        ans = { 'fraction': 100, 'data': _ans/_fy**3}
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_f, _x, _y, _fx, _fy, _fxx, _fyy, _fxy] = quiz.data
        _num = sympy.expand(2*_fx*_fy*_fxy-_fx*_fx*_fyy-_fy*_fy*_fxx)
        if _num != 0:
            ans['feedback'] = r'資料や教科書を確認しましょう。符号を取り違えている可能性があります。'
            ans['data'] = -_num/_fy**3
            answers.append(dict(ans))
        if _fx != _fy:
            ans['feedback'] = r'資料や教科書を確認しましょう。偏導関数の一部を取り違えている可能性があります。'
            ans['data'] = _num/_fx**3
            answers.append(dict(ans))
        if _num != 0 and _fx != _fy:
            ans['feedback'] = r'資料や教科書を確認しましょう。偏導関数の一部や符号を取り違えている可能性があります。'
            ans['data'] = -_num/_fx**3
            answers.append(dict(ans))
        _incorrect_num = sympy.expand(2*_fx*_fy*_fxy+_fx*_fx*_fyy+_fy*_fy*_fxx)
        if _num != _incorrect_num:
            ans['feedback'] = r'資料や教科書を確認しましょう。符号を取り違えている可能性があります。'
            ans['data'] = _incorrect_num/_fy**3
            answers.append(dict(ans))
        if _num != -_incorrect_num:
            ans['feedback'] = r'資料や教科書を確認しましょう。符号を取り違えている可能性があります。'
            ans['data'] = -_incorrect_num/_fy**3
            answers.append(dict(ans))
        if sympy.expand(_num*_fx**3-_incorrect_num*_fy**3) != 0:
            ans['feedback'] = r'資料や教科書を確認しましょう。色々と取り違えている可能性があります。'
            ans['data'] = _incorrect_num/_fx**3
            answers.append(dict(ans))
        if sympy.expand(_num*_fx**3+_incorrect_num*_fy**3) != 0:
            ans['feedback'] = r'資料や教科書を確認しましょう。色々と取り違えている可能性があります。'
            ans['data'] = -_incorrect_num/_fx**3
            answers.append(dict(ans))            
        answers = common.answer_union(answers)
        while len(answers) < size:
            ans['feedback'] = r'資料や教科書を確認しましょう。色々と取り違えている可能性があります。'
            ans['data'] = random.randint(self.elem_min, self.elem_max)
            answers.append(dict(ans))   
            answers = common.answer_union(answers)            
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_f, _x, _y, _fx, _fy, _fxx, _fyy, _fxy] = quiz.data
        _text = r'次式で定義される陰関数\( f(x,y)=0 \)に関して，その2階導関数\( \frac{d^2y}{dx^2} \)を求めてください。<br />'
        _text += r'\( f(x,y)=' + sympy.latex(_f) + r'=0 \)'
        return _text
    def answer_text(self, ans):
        return r'\( ' + sympy.latex(ans['data'], order='lex') + r' \)'


# In[269]:


if __name__ == "__main__":
    q = second_derivative_of_implicit_function()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=25)


# In[270]:


if __name__ == "__main__":
    pass
    #qz.save('second_derivative_of_implicit_function.xml')


# ## extremum condition of implicit function

# In[277]:


class extremum_condition_of_implicit_function(core.Question):
    name = '陰関数が極値を取るための条件'
    def __init__(self, emin=-10, emax=10):
        # 生成する個々の係数等の範囲
        self.elem_min = emin
        self.elem_max = emax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='陰関数の極値の条件', quiz_number=_quiz_number)
        _a = random.randint(self.elem_min, self.elem_max)
        _b = random.randint(self.elem_min, self.elem_max)
        _fx = random.choice([0,0,0,0,0,1])*random.randint(self.elem_min, self.elem_max)
        _fy = random.choice([1,1,1,1,1,0])*random.randint(self.elem_min, self.elem_max)
        _fxx = nonzero_randint(self.elem_min, self.elem_max)
        quiz.quiz_identifier = hash(str(_a) + str(_b) + str(_fx) + str(_fy) + str(_fxx))
        # 正答の選択肢の生成
        if _fy == 0:
            _ans = 'nodef'
        elif _fx != 0:
            _ans = 'no'
        elif _fxx*_fy < 0:
            _ans = 'min'
        else:
            _ans = 'max'
        quiz.data = [_a, _b, _fx, _fy, _fxx, _ans]
        ans = { 'fraction': 100, 'data': [_a, _b, _ans]}
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # 正答を個別には作らないので何もしない
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_a, _b, _fx, _fy, _fxx, _ans] = quiz.data
        ans['feedback'] = r'資料や教科書で極値を取るための陰関数の条件を確認してください。'
        if _ans != 'no':
            ans['data'] = [_a, _b, 'no']
            answers.append(dict(ans))
        if _ans != 'min':
            ans['data'] = [_a, _b, 'min']
            answers.append(dict(ans))
        if _ans != 'max':
            ans['data'] = [_a, _b, 'max']
            answers.append(dict(ans))
        if _ans != 'nodef':
            ans['data'] = [_a, _b, 'nodef']
            answers.append(dict(ans))
        answers = common.answer_union(answers)
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_a, _b, _fx, _fy, _fxx, _ans] = quiz.data
        _text = r'ある陰関数\( f(x,y)=0 \)は，点\( (a,b)=(' + sympy.latex(_a) + r',' + sympy.latex(_b) + r') \)において，'
        _text += r'それぞれ次のような条件を満たしています。極値に関して最も適切な選択肢を選んでください。<br />'
        _text += r'\( f(a,b)=0,\;'
        _text += r'f_{x}(a,b)=' + sympy.latex(_fx) + r',\;'
        _text += r'f_{y}(a,b)=' + sympy.latex(_fy) + r',\;'
        _text += r'f_{xx}(a,b)=' + sympy.latex(_fxx) + r' \)'
        return _text
    def answer_text(self, ans):
        [_a, _b, _ans] = ans['data']
        _text = r'陰関数\( f(x,y) \)は点\( (a,b)=(' + sympy.latex(_a) + r',' + sympy.latex(_b) + r') \)で，'
        if _ans == 'no':
            _text += r'極値を取らない。'
        elif _ans == 'max':
            _text += r'極大値を取る。'
        elif _ans == 'min':
            _text += r'極小値を取る。'
        else:
            _text += r'陰関数定理の条件を満たさない。'
        return _text


# In[278]:


if __name__ == "__main__":
    q = extremum_condition_of_implicit_function()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=25)


# In[279]:


if __name__ == "__main__":
    pass
    #qz.save('extremum_condition_of_implicit_function.xml')


# ## dummy

# In[ ]:





# # All the questions

# In[ ]:


questions_str = ['limit_of_bivariate_polynomial', 'limit_of_bivariate_rational_function_approaches_domain', 
                 'limit_of_bivariate_rational_function_approaches_domain_with_common_factor', 
                 'limit_of_bivariate_rational_function_approaches_boundary', 'limit_with_sin_or_log_over_xpy',
                 'partial_derivative', 'partial_higher_derivative', 'extremum_condition', 'extremum_of_polynomial', 
                 'Lagrangian_multipliers', 'derivative_of_implicit_function', 'second_derivative_of_implicit_function', 
                 'extremum_condition_of_implicit_function']
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




