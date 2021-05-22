import moodle_xqg.core as mxqg
import moodle_xqg.qbank.common as mxqg_common
from prettytable import PrettyTable
import itertools
import random
import math


def overline(text: str) -> str:
    return f'<SPAN STYLE="text-decoration:overline">{text}</SPAN>'


def matrix_to_html_table(matrix: list, n: int) -> str:
    x = PrettyTable(border=True, header=True, hrules=1, vrules=1)
    x.format = True
    headers = list()
    for i in range(1, n+1):
        headers.append(f"x{i}")
    headers.append(f"F({','.join(headers)})")
    x.field_names = headers
    for r in matrix:
        x.add_row(r)
    return x.get_html_string()
    # return tabulate(matrix.tolist(), headers=headers, tablefmt="html")


def matrix_to_function(matrix: list, n: int) -> str:
    minterm = list()
    for r in matrix:
        result = r[n]
        if result == 1:
            variables = list()
            for i, v in enumerate(r[:-1]):
                variable = f"x{i + 1}"
                if v == 0:
                    variable = overline(variable)
                variables.append(variable)
            minterm.append("&nbsp;".join(variables))

    return f'<b>{gen_function(n)}</b> = {" &#43; ".join(minterm)}'


def gen_function(n: int) -> str:
    headers = list()
    for i in range(1, n+1):
        headers.append(f"x{i}")
    return f"F({','.join(headers)})"


def gen_matrix(n: int) -> [[int]]:
    matrix = [list(i) for i in itertools.product([0, 1], repeat=n)]
    for i, r in enumerate(matrix):
        r.append(random.randint(0, 1))
    return matrix


def gen_bad_answer(n: int, correct_answer: str, num_answer: int = 3) -> [str]:
    answers = list()
    while num_answer > 0:
        num_products = random.randint(1, int(math.pow(2, n)))
        minterm = list()
        for j in range(1, num_products+1):
            r = [random.randint(0, 1) for i in range(0, n)]
            variables = list()
            for i, v in enumerate(r):
                variable = f"x{i + 1}"
                if v == 0:
                    variable = overline(variable)
                variables.append(variable)
            minterm.append("&nbsp;".join(variables))
        answer = f'<b>{gen_function(n)}</b> = {" &#43; ".join(minterm)}'
        if answer != correct_answer:
            answers.append(answer)
            num_answer -= 1

    return answers


class BooleanFunction(mxqg.Question):
    n=3

    def question_generate(self, _quiz_number=0):
        quiz = mxqg.Quiz(name='Da tabella di verità a funzione booleana', quiz_number=_quiz_number, lang='it')
        # generates a quiz data
        quiz.data = [gen_matrix(self.n)]
        quiz.quiz_identifier = hash(matrix_to_function(quiz.data[0], self.n))

        return quiz

    def correct_answers_generate(self, quiz, size=1):
        answers = []
        # generates the correct answer
        # generates the correct choice
        correct_answer = matrix_to_function(quiz.data[0], self.n)
        ans = {'fraction': 100, 'data': correct_answer, 'feedback': 'Ottimo!'}
        # quiz.answers.append(ans)
        answers.append(ans)
        answers = mxqg_common.answer_union(answers)
        return answers

    def incorrect_answers_generate(self, quiz, size=4):
        # generates incorrect choices randomly (simple random generation is not recommended though)
        answers = []
        ans = {'fraction': 0, 'data': 0}
        bad_answers = gen_bad_answer(n=self.n, correct_answer=matrix_to_function(quiz.data[0], self.n), num_answer=size)
        for answer in bad_answers:
            ans['data'] = answer
            ans['feedback'] = "Risposta sbagliata! Ricordati che la soluzione corretta è l'OR dei MINTERM "
            answers.append(dict(ans))
            answers = mxqg_common.answer_union(answers)
        return answers

    def question_text(self, quiz):
        return f'<p>Indicare quale delle seguenti funzioni booleane è equivalente a questa tavola di verità</p>' \
               f'<p>{matrix_to_html_table(quiz.data[0], self.n)}</p>'

    def answer_text(self, ans):
        return ans['data']