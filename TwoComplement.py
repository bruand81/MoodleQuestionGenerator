import moodle_xqg.core as mxqg
import moodle_xqg.qbank.common as mxqg_common
import random


class BinaryConversion:

    @staticmethod
    def binary_to_decimal(n: str) -> int:
        return int(n, 2)

    @staticmethod
    def decimal_to_binary(n: int) -> str:
        return bin(n).replace("0b", "")

    @staticmethod
    def complement_binary(binary: str) -> str:
        return binary.replace("0", "x").replace("1", "0").replace("x", "1")

    @staticmethod
    def c2binary_to_decimal(binary: str) -> int:
        e = len(binary) - 1
        msb = int(binary[0])
        lsbs = BinaryConversion.binary_to_decimal(binary[1:])
        sign = -1 * msb * pow(2, e)
        result = sign + lsbs
        return result

    @staticmethod
    def decimal_to_c2binary(n: int, max_bit: int) -> str:
        binary = BinaryConversion.decimal_to_binary(abs(n))
        if len(binary) > (max_bit - 1):
            raise ValueError(f"{n} is too big to be represented with {max_bit} bit")
        binary = ("0" * ((max_bit - 1) - len(binary))) + binary
        binary = BinaryConversion.complement_binary(binary)
        binary = BinaryConversion.sum_one_to_binary(binary)
        return "1" + binary

    @staticmethod
    def sum_one_to_binary(binary: str) -> str:
        bits = [char for char in binary]
        return_bits = list()
        bits.reverse()
        is_first = True
        is_report = False
        for bit in bits:
            if is_first | is_report:
                if bit == "1":
                    bit = "0"
                    is_report = True
                else:
                    bit = "1"
                    is_report = False
                is_first = False
            return_bits.append(bit)
        return_bits.reverse()
        return "".join(return_bits)


class TwoComplementToDecimal(mxqg.Question):
    bit_number = 8

    def question_generate(self, _quiz_number=0):
        quiz = mxqg.Quiz(name='Da complemento a 2 a decimale', quiz_number=_quiz_number, lang='it')
        # generates a quiz data
        quiz.data = ["".join(["1"] + [f"{random.choice([0, 1])}" for i in range(self.bit_number - 1)])]
        quiz.quiz_identifier = hash(quiz.data[0])

        correct_answer = f"{BinaryConversion.c2binary_to_decimal(quiz.data[0])}"
        quiz.data.append(correct_answer)

        return quiz

    def correct_answers_generate(self, quiz, size=1):
        answers = []
        # generates the correct answer
        # generates the correct choice
        ans = {'fraction': 100, 'data': f"{quiz.data[1]}<sub>10</sub>", 'feedback': 'Ottimo!'}
        # quiz.answers.append(ans)
        answers.append(ans)
        answers = mxqg_common.answer_union(answers)
        return answers

    def incorrect_answers_generate(self, quiz, size=4):
        # generates incorrect choices randomly (simple random generation is not recommended though)
        answers = []
        ans = {'fraction': 0, 'data': 0}
        while len(answers) < size:
            incorrect_ans = f"{random.randint(-256, 256)}"
            ans['data'] = f"{incorrect_ans}<sub>10</sub>"
            ans[
                'feedback'] = 'No, ricorda che si calcola come -(valore decimale del MSB) + (valore decimale del ' \
                              'resto della stringa binaria) '
            if incorrect_ans != quiz.data[1]:
                answers.append(dict(ans))  # you may need to copy the object if reuse it
            answers = mxqg_common.answer_union(answers)
        return answers

    def question_text(self, quiz):
        return f'Indicare quale fra i seguenti numeri rappresentati in <u>sistema decimale</u> ' \
               f'corrisponde al numero {quiz.data[0]}<sub>C2</sub> espresso in <u>complemento a due ' \
               f'a {self.bit_number} bit</u>'

    def answer_text(self, ans):
        return ans['data']


class DecimalToTwoComplement(mxqg.Question):
    bit_number = 8

    def question_generate(self, _quiz_number=0):
        quiz = mxqg.Quiz(name='Da decimale a complemento a 2', quiz_number=_quiz_number, lang='it')
        # generates a quiz data
        target_value = random.randint(-1 * (pow(2, self.bit_number - 1)-1), -1)
        quiz.data = [f"{target_value}"]
        quiz.quiz_identifier = hash(quiz.data[0])
        correct_answer = BinaryConversion.decimal_to_c2binary(target_value, self.bit_number)
        quiz.data.append(correct_answer)
        return quiz

    def correct_answers_generate(self, quiz, size=1):
        answers = []
        ans = {'fraction': 100, 'data': f"{quiz.data[1]}<sub>C2</sub>", 'feedback': 'Ottimo!'}
        # quiz.answers.append(ans)
        answers.append(ans)
        answers = mxqg_common.answer_union(answers)
        return answers

    def incorrect_answers_generate(self, quiz, size=4):
        # generates incorrect choices randomly (simple random generation is not recommended though)
        correct_answer = int(quiz.data[0])
        answers = []
        ans = {'fraction': 0, 'data': 0}
        while len(answers) < size:
            generated_value = random.randint(-1 * (pow(2, self.bit_number - 1)-1), -1)
            ans['data'] = f"{BinaryConversion.decimal_to_c2binary(generated_value, self.bit_number)}<sub>C2</sub>"
            ans['feedback'] = f'No, ricorda che si calcola con il seguente algoritmo:</br>' \
                              f'<ol>' \
                              f'<li>Converti il valore assoluto di {correct_answer} in binario semplice</li>' \
                              f'<li>Somma 1 <u>(in binario)</u> a questo numero</li>' \
                              f'<li>Aggiungi 1 come MSB (Most Significant Bit) al valore ottenuto</li>' \
                              f'</ol>'
            if generated_value != correct_answer:
                answers.append(dict(ans))  # you may need to copy the object if reuse it
            answers = mxqg_common.answer_union(answers)
        return answers

    def question_text(self, quiz):
        return f"Indicare quale fra i seguenti numeri rappresentati in " \
               f"<u>sistema binario in complemento a due</u> (su {self.bit_number} bit) " \
               f"corrisponde al numero {quiz.data[0]} espresso " \
               f"in <u>sistema decimale</u> ({quiz.data[0]}<sub>10</sub>)"

    def answer_text(self, ans):
        return ans['data']
