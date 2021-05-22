#!/usr/bin/env python
# coding: utf-8

# Moodle XML Question Generator, (C) 2020 Kosaku Nagasaka (Kobe University)

# note
# > ***This notebook includes only some English ported generators (only 2 of more than 100).***

# ## generate the module file

# In[19]:


if __name__ == "__main__":
    import subprocess
    subprocess.run(['jupyter', 'nbconvert', '--to', 'python', '_basic_calculus_in_english.ipynb','--output','basic_calculus_in_english.py'])


# # Basic Calculus

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
else:
    from .. import core
    from . import common


# In[2]:


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


# In[7]:


def flatten_list(alist):
    rlist = []
    for lis in alist:
        rlist = rlist + lis
    return rlist


# In[8]:


def flatten_list_all(alist):
    if type(alist) is not list:
        return [alist]
    rlist = []
    for lis in alist:
        rlist = rlist + flatten_list_all(lis)
    return rlist


# ## power series expansion (en)

# In[11]:


class power_series_expansion(core.Question):
    name = 'power series expansion (Taylor and Maclaurin serieses)'
    _x = sympy.Symbol('x')
    def __init__(self, emin=-3, emax=3, nmin=2, nmax=4, amin=-2, amax=2, azero=True, omin=1, omax=3):
        # range of the elements
        self.elem_min = emin
        self.elem_max = emax
        # range of the expansion point
        self.epnt_min = amin
        self.epnt_max = amax
        self.epnt_zero = azero
        # range of the degree of polynomials
        self.deg_min = nmin
        self.deg_max = nmax
        # range of the expansion order
        self.ord_min = omin
        self.ord_max = omax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='power series expansion', quiz_number=_quiz_number)
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
        # generate the answer
        _series = _func.series(x=self._x, x0=_point, n=_order+1).removeO()        
        quiz.data = [_func, _point, _order, _series]
        ans = { 'fraction': 100, 'data': _series }
        quiz.answers.append(ans)
        return quiz        
    def correct_answers_generate(self, quiz, size=1):
        # do nothing since we have generated the answer
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
                ans['feedback'] = r'take the first several terms of the Taylor expansion to make the sum a polynomial of the specified degree.'
                ans['data'] = _incorrect
                answers.append(dict(ans))
            _incorrect = self._incorrect_series_expansion_by_factorial(_func, _point, _n)   
            if _incorrect != _series:
                ans['feedback'] = r'check the definition of the Taylor series. especially take of the factorial part.'
                ans['data'] = _incorrect
                answers.append(dict(ans))
            _incorrect = self._incorrect_series_expansion_by_factorial2(_func, _point, _n)   
            if _incorrect != _series:
                ans['feedback'] = r'check the definition of the Taylor series. especially take of the factorial part.'
                ans['data'] = _incorrect
                answers.append(dict(ans))
        if _point != 0:
            for _n in range(2,_order+2):
                _incorrect = self._incorrect_series_expansion_by_signofa(_func, _point, _n)   
                if _incorrect != _series:
                    ans['feedback'] = r'check the definition of the Taylor series. especially take of the expansion point.'
                    ans['data'] = _incorrect
                    answers.append(dict(ans))
                _incorrect = self._incorrect_series_expansion_by_factorial_and_signofa(_func, _point, _n)   
                if _incorrect != _series:
                    ans['feedback'] = r'check the definition of the Taylor series. your understandings are far from the definition.'
                    ans['data'] = _incorrect
                    answers.append(dict(ans))
                _incorrect = self._incorrect_series_expansion_by_factorial2_and_signofa(_func, _point, _n)   
                if _incorrect != _series:
                    ans['feedback'] = r'check the definition of the Taylor series. your understandings are far from the definition.'
                    ans['data'] = _incorrect
                    answers.append(dict(ans))
        answers = common.answer_union(answers)
        while len(answers) < size:
            _incorrect = _series + random.randint(self.elem_min, self.elem_max)
            if _incorrect != _series:
                ans['feedback'] = r'check the definition of the Taylor series.'
                ans['data'] = _incorrect
                answers.append(dict(ans))
        if len(answers) >= size:
            return random.sample(answers,k=size)
        return answers    
    def question_text(self, quiz):
        [_func, _point, _order, _series] = quiz.data
        _text = r'Choose the approximation as a polynomial of degree \( ' + sympy.latex(_order) + r' \) '
        if _point == 0:
            _text += r'(based on Maclaurin series)'
        else:
            _text += r'(based on Taylor series at \( x=' + sympy.latex(_point) + r' \))'
        _text += r' of the following function \( f(x) \). <br />'
        _text += r'\( f(x)=' + sympy.latex(_func,  order='lex') + r' \)'
        return _text
    def answer_text(self, ans):
        return r'\( ' +  sympy.latex(ans['data'], order='lex', long_frac_ratio=1) + r' \)'


# In[12]:


if __name__ == "__main__":
    q = power_series_expansion(emin=-4, emax=4, amin=0, amax=0)
    qz = core.generate(q, size=200, category=r'Maclaurin expansion')
    qz.preview(size=25)


# In[35]:


if __name__ == "__main__":
    #pass
    qz.save('power_series_expansion_maclaurin_in_english.xml')


# ## extremum condition of bivariate function (en)

# In[17]:


class extremum_condition_bivariate_function(core.Question):
    name = 'extremum condition of bivariate function'
    def __init__(self, emin=-10, emax=10):
        # range of the elements
        self.elem_min = emin
        self.elem_max = emax
    def question_generate(self, _quiz_number=0):
        quiz = core.Quiz(name='extremum condition', quiz_number=_quiz_number)
        _a = random.randint(self.elem_min, self.elem_max)
        _b = random.randint(self.elem_min, self.elem_max)
        _fx = random.choice([0,0,0,0,0,1])*random.randint(self.elem_min, self.elem_max)
        _fy = random.choice([0,0,0,0,0,1])*random.randint(self.elem_min, self.elem_max)
        _fxx = nonzero_randint(self.elem_min, self.elem_max)
        _fyy = random.randint(self.elem_min, self.elem_max)
        _fxy = random.randint(sympy.floor(self.elem_min/5), sympy.ceiling(self.elem_max/5))
        _dxy = _fxx*_fyy-_fxy**2
        quiz.quiz_identifier = hash(str(_a) + str(_b) + str(_fx) + str(_fy) + str(_fxx) + str(_fyy) + str(_fxy) + str(_dxy))
        # generate the answer
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
        # do nothing since we have already generated the answer
        pass
    def incorrect_answers_generate(self, quiz, size=4):
        answers = []
        ans = { 'fraction': 0 }
        [_a, _b, _fx, _fy, _fxx, _fyy, _fxy, _dxy, _ans] = quiz.data
        ans['feedback'] = r'check whether the necessary and sufficient conditions are satisfied or not.'
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
        _text = r'\( f(x,y) \) is a \( C^2 \) function and satisfies the following conditions at \( (a,b)=(' + sympy.latex(_a) + r',' + sympy.latex(_b) + r') \)'
        _text += r'. Choose the best one in terms of extremum of \( f(x,y) \) at \( (a,b) \).<br />'
        _text += r'\( f_{x}(a,b)=' + sympy.latex(_fx) + r',\;'
        _text += r'f_{y}(a,b)=' + sympy.latex(_fy) + r',\;'
        _text += r'f_{xx}(a,b)=' + sympy.latex(_fxx) + r',\;'
        _text += r'f_{yy}(a,b)=' + sympy.latex(_fyy) + r',\;'
        _text += r'f_{xy}(a,b)=' + sympy.latex(_fxy) + r' \)'
        return _text
    def answer_text(self, ans):
        [_a, _b, _ans] = ans['data']
        if _ans == 'no':
            _text = r'The function \( f(x,y) \) ' + r'does not take any extremum'
        elif _ans == 'max':
            _text = r'The function \( f(x,y) \) ' + r'takes a local maximum'
        elif _ans == 'min':
            _text = r'The function \( f(x,y) \) ' + r'takes a local minimum'
        else:
            _text = r'With only the information given, it is undecidable whether the function \( f(x,y) \) takes any extremum or not '
        _text += r' at \( (a,b)=(' + sympy.latex(_a) + r',' + sympy.latex(_b) + r') \).'
        return _text


# In[18]:


if __name__ == "__main__":
    q = extremum_condition_bivariate_function()
    qz = core.generate(q, size=200, category=q.name)
    qz.preview(size=25)


# In[7]:


if __name__ == "__main__":
    #pass
    qz.save('extremum_condition_bivariate_function_in_english.xml')


# ## dummy

# In[ ]:





# # All the questions

# In[ ]:


questions_str = ['power_series_expansion', 'extremum_condition_bivariate_function']
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




