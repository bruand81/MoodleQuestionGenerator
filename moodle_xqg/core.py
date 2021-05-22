#!/usr/bin/env python
# coding: utf-8

# Moodle XML Question Generator, (C) 2019-2020 Kosaku Nagasaka (Kobe University)

# # Moodle XML Question Generator Core API

# In[ ]:


"""Module moodle_xqg.core -- helpers for generating multichoice quizzes for Moodle.   

Class Quiz:
    For a quiz instance generated by core.generate with core.Question.
    
Class Quizzes:
    For a set of quizzes generated by core.generate with core.Question.
    
Class Question:
    For a base class for each question specification/generator.
"""


# ## Required modules

# In[148]:


import sympy
import pickle
import lxml.etree
import random
import IPython
import time


# ## Common subroutines

# In[ ]:


_force_to_this_lang = None


# In[ ]:


locale_dict = dict()
locale_dict['ja'] = dict()
locale_dict['ja']['('] = r'（'
locale_dict['ja'][')'] = r'）'
locale_dict['ja']['Quiz ID'] = r'問題整理番号'
locale_dict['ja']['Feedback'] = r'フィードバック'
locale_dict['ja']['General Feedback'] = r'フィードバック'
locale_dict['ja']['Correct Feedback'] = r'正答フィードバック'
locale_dict['ja']['Partially Correct Feedback'] = r'部分フィードバック'
locale_dict['ja']['Incorrect Feedback'] = r'誤答フィードバック'
locale_dict['ja']['Score'] = r'点数'
locale_dict['ja']['Your answer is correct.'] = r'あなたの答えは正解です。'
locale_dict['ja']['Your answer is partially correct.'] = r'あたなの答えは正しくありません。'
locale_dict['ja']['Your answer is incorrect.'] = r'あたなの答えは部分的に正解です。'
locale_dict['it'] = dict()
locale_dict['it']['('] = r'（'
locale_dict['it'][')'] = r'）'
locale_dict['it']['Quiz ID'] = r'Quiz ID'
locale_dict['it']['Feedback'] = r'Feedback'
locale_dict['it']['General Feedback'] = r'General Feedback'
locale_dict['it']['Correct Feedback'] = r'Correct Feedback'
locale_dict['it']['Partially Correct Feedback'] = r'Partially Correct Feedback'
locale_dict['it']['Incorrect Feedback'] = r'Incorrect Feedback'
locale_dict['it']['Score'] = r'Score'
locale_dict['it']['Your answer is correct.'] = r'La tua risposta è corretta'
locale_dict['it']['Your answer is partially correct.'] = r'La tua risposta è sbagliata'
locale_dict['it']['Your answer is incorrect.'] = r'La tua risposta è sbagliata'


# In[1]:


def _ldt(text, lang):
    if _force_to_this_lang is not None:
        lang = _force_to_this_lang
    if lang in locale_dict.keys():
        if text in locale_dict[lang].keys():
            return locale_dict[lang][text]
    return text


# In[ ]:


def _text_translate(text, dic):
    for key,val in dic.items():
        text = text.replace(key,val)
    return text


# In[ ]:


def _append_text_tag(parent, text, dic=dict()):
    target_txt = lxml.etree.SubElement(parent, 'text')
    target_txt.text = _text_translate(text, dic)


# In[ ]:


def _append_cdata_tag(parent, text, dic=dict()):
    target_txt = lxml.etree.SubElement(parent, 'text')
    target_txt.text = lxml.etree.CDATA(_text_translate(text, dic))


# In[ ]:


def _append_category_tag(parent, category, dic=dict()):
    question_tag = lxml.etree.SubElement(parent, 'question', {'type':'category'})
    category_tag = lxml.etree.SubElement(question_tag, 'category')
    _append_text_tag(category_tag, category, dic)


# In[25]:


def _append_answer_tag(parent, answer, dic=dict()):
    answer_tag = lxml.etree.SubElement(parent, 'answer', {'fraction':str(answer['fraction']), 'format':'html'})
    _append_cdata_tag(answer_tag, answer['text'], dic)
    if 'feedback' in answer:
        feedback_tag = lxml.etree.SubElement(answer_tag, 'feedback', {'format':'html'})
        _append_text_tag(feedback_tag, answer['feedback'], dic)


# In[ ]:


def _append_question_tag(parent, quiz, show_quiz_number=True, show_quiz_number_adminonly=False, dic=dict(), lang='en'):
    question_tag = lxml.etree.SubElement(parent, 'question', {'type':'multichoice'})
    name_tag = lxml.etree.SubElement(question_tag, 'name')
    if show_quiz_number:
        _append_text_tag(name_tag, quiz.name + ' (id:{:03})'.format(quiz.quiz_number), dic)
    else:
        _append_text_tag(name_tag, quiz.name, dic)
    questiontext_tag = lxml.etree.SubElement(question_tag, 'questiontext', {'format':'html'})
    if show_quiz_number and not show_quiz_number_adminonly:
        _append_cdata_tag(questiontext_tag, quiz.question_text +                           '<div style="text-align:right;font-size:smaller">' +                           _ldt('(',lang) + _ldt('Quiz ID',lang) + ':{:03}'.format(quiz.quiz_number) +                           _ldt(')',lang) + '</div>', dic)
    else:
        _append_cdata_tag(questiontext_tag, quiz.question_text, dic)
    general_feedback_tag = lxml.etree.SubElement(question_tag, 'generalfeedback', {'format':'html'})
    _append_text_tag(general_feedback_tag, quiz.general_feedback, dic)
    default_grade_tag = lxml.etree.SubElement(question_tag, 'defaultgrade')
    default_grade_tag.text = str(quiz.default_grade)    
    penalty_tag = lxml.etree.SubElement(question_tag, 'penalty')
    penalty_tag.text = str(quiz.penalty)
    hidden_tag = lxml.etree.SubElement(question_tag, 'hidden')
    hidden_tag.text = '1' if quiz.hidden else '0'
    single_tag = lxml.etree.SubElement(question_tag, 'single')
    single_tag.text = 'true' if quiz.single else 'false'
    shuffle_answers_tag = lxml.etree.SubElement(question_tag, 'shuffleanswers')
    shuffle_answers_tag.text = 'true' if quiz.shuffle_answers else 'false'
    answer_numbering_tag = lxml.etree.SubElement(question_tag, 'answernumbering')
    answer_numbering_tag.text = quiz.answer_numbering 
    correct_feedback_tag = lxml.etree.SubElement(question_tag, 'correctfeedback', {'format':'html'})
    _append_cdata_tag(correct_feedback_tag, quiz.correct_feedback, dic)
    partially_correct_feedback_tag = lxml.etree.SubElement(question_tag, 'partiallycorrectfeedback', {'format':'html'})
    _append_cdata_tag(partially_correct_feedback_tag, quiz.partially_correct_feedback, dic)
    incorrect_feedback_tag = lxml.etree.SubElement(question_tag, 'incorrectfeedback', {'format':'html'})
    _append_cdata_tag(incorrect_feedback_tag, quiz.incorrect_feedback, dic)
    for ans in quiz.answers:
        _append_answer_tag(question_tag, ans, dic)


# In[65]:


def _generate_question_html(quiz, show_quiz_number=True, show_quiz_number_adminonly=False, dic=dict(), lang='en'):
    fullhtml = '<div style="border: thin solid #000000; padding: 1em">'
    feedback_for_question = _ldt('General Feedback',lang) + ':'+_text_translate(quiz.general_feedback, dic)
    feedback_for_question += '\n' + _ldt('Correct Feedback',lang) + ':'+_text_translate(quiz.correct_feedback, dic)
    feedback_for_question += '\n' + _ldt('Partially Correct Feedback',lang) + ':'+_text_translate(quiz.partially_correct_feedback, dic)
    feedback_for_question += '\n' + _ldt('Incorrect Feedback',lang) + ':'+_text_translate(quiz.incorrect_feedback, dic)
    fullhtml += '<div title="' + feedback_for_question + '">'
    fullhtml += _text_translate(quiz.question_text, dic)
    if show_quiz_number and not show_quiz_number_adminonly:
        fullhtml += _text_translate('<div style="text-align:right;font-size:smaller">' +                                     _ldt('(',lang) + _ldt('Quiz ID',lang) + ':{:03}'.format(quiz.quiz_number) +                                     _ldt(')',lang) + '</div>', dic)
    answer_ids = list(range(len(quiz.answers)))
    if quiz.shuffle_answers:
        answer_ids = random.sample(answer_ids, len(answer_ids))
    list_style = 'list-style-type:none'
    if quiz.answer_numbering == 'abc':
        list_style = 'list-style-type:lower-latin'
    elif quiz.answer_numbering == 'ABC':
        list_style = 'list-style-type:upper-latin'
    elif quiz.answer_numbering == '123':
        list_style = 'list-style-type:decimal'
    fullhtml += '<ol style="' + list_style +'">'
    for ans_id in answer_ids:
        attr_li = _ldt('Score',lang) + ':' + str(quiz.answers[ans_id]['fraction'])
        if 'feedback' in quiz.answers[ans_id]:
            attr_li += '\n' + _ldt('Feedback',lang) + ':' + _text_translate(quiz.answers[ans_id]['feedback'], dic)
        fullhtml += '<li title="' + attr_li + '">'
        if quiz.single:
            fullhtml += '<input type="radio" name="preview' + str(quiz.quiz_number) + '">'
        else:
            fullhtml += '<input type="checkbox" name="preview' + str(quiz.quiz_number) + '">'
        fullhtml += _text_translate(quiz.answers[ans_id]['text'], dic)
        fullhtml += '</li>'    
    fullhtml += '</ol>' # ol
    fullhtml += '</div>' # question   
    fullhtml += '</div>'
    return fullhtml


# ## Class definition for a quiz

# In[7]:


class Quiz:
    """Quiz is a class representing a generated quiz. 
    
    Attributes:
        category(str): category name of moodle quiz module. 
        name(str): quiz name that will be appeared in the list of quiz module. 
        general_feedback(str): general feedback text.
        default_grade(float): default grade option for moodle XML format.
        penalty(float): penalty option for moodle XML format.
        hidden(boolean): hidden option for moodle XML format.
        single(boolean): single option for moodle XML format.
        shuffle_answers(boolean): shuffle_answers option for moodle XML format.
        answer_numbering(str): 'none', 'abc', 'ABC' or '123' for numbering.
        correct_feedback(str): correct feedback text.
        incorrect_feedback(str): incorrect feedback text.
        partially_correct_feedback(str): partially correct feedback text.
        data(any): internal data store while generating a quiz.
        answers(list): a list of answers where each answer is a dictionaly: 
                        {'fraction':-, 'data':-, 'feedback':-, 'text':- }
        question_text(str): the final generated question text.
        quiz_number(int): identifier number to distinguish the quiz for person.
        quiz_identifier(any): identifier to distinguish the quiz during the generation.
    """
    def __init__(self, category='', name='', general_feedback='',                  default_grade=1.0, penalty=0.0, hidden=False,                  single=True, shuffle_answers=True,                  answer_numbering='none', quiz_number=0,                  correct_feedback='Your answer is correct.',                  incorrect_feedback='Your answer is partially correct.',                  partially_correct_feedback='Your answer is incorrect.', lang='en'):
        self.lang = lang
        self.category = category
        self.name = name
        self.general_feedback = general_feedback
        self.default_grade = default_grade
        self.penalty = penalty
        self.hidden = hidden
        self.single = single
        self.shuffle_answers = shuffle_answers
        self.answer_numbering = answer_numbering
        self.correct_feedback = _ldt(correct_feedback,lang)
        self.incorrect_feedback = _ldt(incorrect_feedback,lang)
        self.partially_correct_feedback = _ldt(partially_correct_feedback,lang)
        self.data = []
        self.question_text = ''
        self.answers = [] # [answer0, answer1, ...]
        # answer = {fraction: , data:, feedback:, text: }
        self.quiz_number = quiz_number
        self.quiz_identifier = 0
    def save(self, filename, append=False, internal=False, translate=dict(), show_quiz_number=True, show_quiz_number_adminonly=False):
        """This saves the quiz into the specified file.
        
        If internal=False, save the quiz into the file in Moodle XML format.
        If internal=True, save the quiz into the file as Python object.
        
        Note:
            append, translate, show_quiz_number and show_quiz_number_adminonly options are only applicable for internal=False.
        """
        if internal == False:
            self._save_xml(filename, append, translate, show_quiz_number, show_quiz_number_adminonly)
        else:
            self._save_pickle(filename)
    def _save_xml(self, filename, append=False, translate=dict(), show_quiz_number=True, show_quiz_number_adminonly=False):
        root = lxml.etree.Element('quiz')
        if len(self.category) > 0:
            _append_category_tag(root, self.category, translate)
        _append_question_tag(root, self, show_quiz_number, show_quiz_number_adminonly, translate, lang=self.lang)
        with open(filename, 'wb') as f:
            f.write(lxml.etree.tostring(root, encoding='UTF-8', xml_declaration=True, pretty_print=True))
    def _save_pickle(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
    def preview(self, translate=dict(), show_quiz_number=True, show_quiz_number_adminonly=False):
        IPython.display.display(IPython.display.HTML(_generate_question_html(self, show_quiz_number, show_quiz_number_adminonly, translate, lang=self.lang)))


# ## Class definition for a set of quizzes

# In[1]:


class Quizzes:
    def __init__(self):
        self.quizzes = []
    def number_of_quizzes(self):
        return len(self.quizzes)
    def append(self, quiz):
        for q in self.quizzes:
            if q.quiz_identifier == quiz.quiz_identifier:
                return False
        self.quizzes.append(quiz)
        return True
    def save(self, filename, append=False, internal=False, translate=dict(), show_quiz_number=True, show_quiz_number_adminonly=False):
        if internal == False:
            self._save_xml(filename, append, translate, show_quiz_number, show_quiz_number_adminonly)
        else:
            self._save_pickle(filename)
    def _save_xml(self, filename, append=False, translate=dict(), show_quiz_number=True, show_quiz_number_adminonly=False):
        root = lxml.etree.Element('quiz')
        if len(self.quizzes) > 0:
            current_category = self.quizzes[0].category
            if len(current_category) > 0:
                _append_category_tag(root, current_category, translate)
            for quiz in self.quizzes:
                if current_category != quiz.category:
                    current_category = quiz.category
                    if len(current_category) > 0:
                        _append_category_tag(root, current_category, translate)
                _append_question_tag(root, quiz, show_quiz_number, show_quiz_number_adminonly, translate, lang=quiz.lang)
        with open(filename, 'wb') as f:
            f.write(lxml.etree.tostring(root, encoding='UTF-8', xml_declaration=True, pretty_print=True))
    def _save_pickle(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
    def preview(self, size=1, shuffle=True, translate=dict(), show_quiz_number=True, show_quiz_number_adminonly=False):
        quiz_ids = list(range(len(self.quizzes)))
        if shuffle:
            quiz_ids = random.sample(quiz_ids, len(quiz_ids))
        fullhtml = ''
        for ids in quiz_ids[:size]:
            fullhtml += _generate_question_html(self.quizzes[ids], show_quiz_number, show_quiz_number_adminonly, translate, lang=self.quizzes[ids].lang)
        IPython.display.display(IPython.display.HTML(fullhtml))
    def listview(self, size=10, translate=dict(), show_quiz_number=True, show_quiz_number_adminonly=False):
        if len(self.quizzes) > 0:
            counter = 0
            current_category = self.quizzes[0].category
            fullhtml = '<ul><li>' + _text_translate(current_category, translate) + '<ul>'
            for quiz in self.quizzes:
                if current_category != quiz.category:
                    current_category = quiz.category
                    fullhtml += '</ul></li><li>' + _text_translate(current_category, translate) + '<ul>'
                if show_quiz_number:
                    fullhtml += '<li>' + _text_translate(quiz.name, translate) + ' (id:{:03})'.format(quiz.quiz_number) + '</li>'
                else:
                    fullhtml += '<li>' + _text_translate(quiz.name, translate) + '</li>'
                counter += 1
                if counter >= size:
                    break;
            fullhtml += '</ul></li></ul>'
            IPython.display.display(IPython.display.HTML(fullhtml))


# ## IO function for loading the saved pickle data

# In[ ]:


def load_internal(filename):
    """This loads Quiz or Quizzes objects from the specified file in the Pickle format."""
    with open(filename, 'rb') as f:
        return pickle.load(f)


# In[137]:


def _load_xml_readtext(parent):
    for child in parent:
        if child.tag == 'text':
            if type(child.text) == str:
                return child.text
            else:
                return ''
    return ''


# In[135]:


def _load_xml_multichoice(parent, current_category):
    quiz = Quiz(category=current_category)
    quiz.quiz_identifier = hash(time.time())
    for child in parent:
        if child.tag == 'name':
            quiz.name = _load_xml_readtext(child)
        elif child.tag == 'questiontext':
            quiz.question_text = _load_xml_readtext(child)
        elif child.tag == 'generalfeedback':
            quiz.general_feedback = _load_xml_readtext(child)
        elif child.tag == 'defaultgrade':
            quiz.default_grade = float(child.text)
        elif child.tag == 'penalty':
            quiz.penalty = float(child.text)
        elif child.tag == 'hidden':
            quiz.hidden = True if child.text == 'true' or child.text == '1' else False            
        elif child.tag == 'single':
            quiz.single = True if child.text == 'true' or child.text == '1' else False
        elif child.tag == 'shuffleanswers':
            quiz.shuffle_answers = True if child.text == 'true' or child.text == '1' else False
        elif child.tag == 'answernumbering':
            quiz.answer_numbering = child.text
        elif child.tag == 'correctfeedback':
            quiz.correct_feedback = _load_xml_readtext(child)
        elif child.tag == 'partiallycorrectfeedback':
            quiz.partially_correct_feedback = _load_xml_readtext(child)
        elif child.tag == 'incorrectfeedback':
            quiz.incorrect_feedback = _load_xml_readtext(child)
        elif child.tag == 'answer':
            ans = dict()
            ans['fraction'] = float(child.attrib['fraction'])
            ans['text'] = _load_xml_readtext(child)
            if len(child) > 1:
                ans['feedback'] = _load_xml_readtext(child[1])
            quiz.answers.append(ans)
    return quiz


# In[146]:


def load_xml(filename):
    """This loads Quiz or Quizzes objects from the specified file in Moodle XML format."""
    with open(filename, 'rb') as f:
        parser = lxml.etree.XMLParser(strip_cdata=False)
        tree = lxml.etree.parse(f, parser=parser)
        root = tree.getroot()
        current_category = ''
        quizzes = Quizzes()
        for child in root:
            if child.tag != 'question':
                continue
            if 'type' not in child.attrib:
                continue
            if child.attrib['type'] == 'category':
                current_category = child[0][0].text
            elif child.attrib['type'] == 'multichoice':
                quizzes.append(_load_xml_multichoice(child, current_category))
        return quizzes


# ## Base Class definition for each quiz generator

# In[8]:


class Question:
    """Question is the base class for user defined questions.

    Define your child class by
    >>> class myQuestion(Question):
    >>>   ...

    Note:
        All the methods should be overridden. The default behaviors are just a meaningless example.
    """
    name = ''
    def __init__(self, lang='en'):
        self.lang = lang
    def question_generate(self, _quiz_number=0):
        return Quiz(quiz_number=_quiz_number, lang=self.lang)
    def correct_answers_generate(self, quiz, size=1):
        ans = dict()
        ans['fraction'] = 100
        return ans
    def incorrect_answers_generate(self, quiz, size=3):
        ans = dict()
        ans['fraction'] = 0
        return ans
    def question_text(self, quiz):
        return 'Choose the correct one.'
    def answer_text(self, ans):
        return 'This is wrong.'


# ## Top level method for generating instances of quiz

# In[ ]:


def _is_quiz_in_quizzes(quiz, quizzes):
    is_found = False
    for q in quizzes.quizzes:
        if q.quiz_identifier == quiz.quiz_identifier:
            is_found = True
            break
    return is_found


# In[ ]:


def _is_ans_in_answers(ans, answers):
    is_found = False
    for a in answers:
        if ans['data'] == a['data']:
            is_found = True
            break
    return is_found


# In[ ]:


def _translate_quiz(quiz, dic):
    quiz.category = _text_translate(quiz.category, dic)
    quiz.name = _text_translate(quiz.name, dic)
    quiz.question_text = _text_translate(quiz.question_text, dic)
    quiz.general_feedback = _text_translate(quiz.general_feedback, dic)
    quiz.correct_feedback = _text_translate(quiz.correct_feedback, dic)
    quiz.partially_correct_feedback = _text_translate(quiz.partially_correct_feedback, dic)
    quiz.incorrect_feedback = _text_translate(quiz.incorrect_feedback, dic)
    for ans in quiz.answers:
        ans['text'] = _text_translate(ans['text'], dic)


# In[11]:


def generate(question_in, size=10, translate=dict(), category=None):
    quizzes = Quizzes()
    if isinstance(question_in, list):
        question_list = question_in
    else:
        question_list = [question_in]
    for question in question_list:
        retry_counter = 0
        size_prev = quizzes.number_of_quizzes()
        while quizzes.number_of_quizzes() - size_prev < size:
            if retry_counter > 256:
                raise RecursionError('quiz generation failed.')
            # generate question
            quiz = question.question_generate(quizzes.number_of_quizzes() - size_prev + 1)
            if _is_quiz_in_quizzes(quiz, quizzes):
                retry_counter += 1
                continue
            quiz.question_text = question.question_text(quiz)
            if category is None:
                quiz.category = question.name
            else:
                quiz.category = category
            # correct answer
            correct_answers = question.correct_answers_generate(quiz)
            if type(correct_answers) == list:
                for ans in correct_answers:
                    if _is_ans_in_answers(ans, quiz.answers):
                        continue;
                    else:
                        quiz.answers.append(ans)
            # incorrect answer
            incorrect_answers = question.incorrect_answers_generate(quiz)
            if type(incorrect_answers) == list:
                for ans in incorrect_answers:
                    if _is_ans_in_answers(ans, quiz.answers):
                        continue;
                    else:
                        quiz.answers.append(ans)
            # answer text
            for ans in quiz.answers:
                ans['text'] = question.answer_text(ans)
            # translate
            _translate_quiz(quiz, translate)
            # add
            quizzes.append(quiz)
    return quizzes


# ## generate the module file

# In[4]:


if __name__ == "__main__":
    import subprocess
    subprocess.run(['jupyter', 'nbconvert', '--to', 'python', '_core.ipynb','--output','core.py'])


# In[ ]:




