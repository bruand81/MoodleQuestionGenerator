#!/usr/bin/env python
# coding: utf-8

# Moodle XML Question Generator, (C) 2019 Kosaku Nagasaka (Kobe University)

# In[4]:


import sympy
import io
import re


# ## general subroutines for internal quiz structure

# In[ ]:


def answer_union(answers):
    """This deletes the duplicate items w.r.t. 'data'."""
    results = []
    for ans in answers:
        is_found = False
        for res in results:
            if ans['data'] == res['data']:
                is_found = True
                break
        if is_found == False:
            results.append(ans)
    return results


# ## conversion from sympy expression to other formats

# In[1]:


def sympy_expr_to_text(expr, prefix='', suffix=''):
    """This converts the given sympy expresion into latex format with the math mode delimiters."""
    return r'\( '+prefix+ sympy.latex(expr, mat_delim='', mat_str='pmatrix', order='lex') +suffix+r' \)'
def sympy_str_to_text(expr_as_str, prefix='', suffix=''):
    """This converts the given sympy expresion as text into latex format with the math mode delimiters.
    Please take care of the limitations of sympy.sympify. For example, '0+1' will be '1'."""
    return r'\( '+prefix+ sympy.latex(sympy.sympify(expr_as_str, evaluate=False), mat_delim='', mat_str='pmatrix', order='lex') +suffix+r' \)'


# In[15]:


def sympy_expr_to_svg(f, **kwargs):
    """This converts the given sympy expression in one variable into svg of the graph y=f(x)."""
    p1 = sympy.plotting.plot(f, show=False, **kwargs)
    svg_data = io.StringIO()
    if hasattr(p1, '_backend'):
        p1._backend.close()
    p1._backend = p1.backend(p1)
    p1._backend.process_series()
    p1._backend.fig.savefig(svg_data, format='svg')
    p1._backend.close() #p1._backend.fig.clf()
    svg_data.seek(0)
    svg_str = svg_data.read()
    svg_str = svg_str[svg_str.find('<svg '):]
    svg_str = re.sub(r'height="(.*?)" ', "", svg_str, 1)
    svg_str = re.sub(r'width="(.*?)" ', "", svg_str, 1)
    return svg_str


# ## generate the module file

# In[1]:


if __name__ == "__main__":
    import subprocess
    subprocess.run(['jupyter', 'nbconvert', '--to', 'python', '_common.ipynb','--output','common.py'])


# In[ ]:




