from typing import Union

import click

from BooleanFunction import BooleanFunction
from TwoComplement import BinaryConversion, TwoComplementToDecimal, DecimalToTwoComplement, BinaryToDecimal, \
    DecimalToBinary
import moodle_xqg.core as mxqg
import moodle_xqg.qbank.common as mxqg_common
import os


# @click.group(invoke_without_command=True)
# @click.pass_context
@click.group()
@click.pass_context
def main(ctx):
    """
    Programma per la generazione di quiz Moodle per il corso di Fondamenti di Informatica
    """
    # if ctx.invoked_subcommand is None:
    #     click.echo('I was invoked without subcommand')
    pass


@main.command("c2decimale", short_help="Da complemento a 2 a decimale",
              help="Comando per la generazione di quiz del tipo "
                   "da complemento a 2 a decimale")
@click.option('-f', '--file_name', default=None, show_default=True, prompt="Inserire il nome del file di destinazione",
              prompt_required=False)
@click.option('-d', '--output_dir', default=None, show_default=True,
              prompt="Inserire il nome della directory di destinazione", prompt_required=False)
@click.option('-c', '--quiz_category', default="Da complemento a 2 a decimale", show_default=True)
@click.option('-s', '--size', type=int, default=50, help="Numero di domande generato", show_default=True,
              prompt="Inserire il numero di domande da generare")
def two_complement_to_decimal(file_name: str, output_dir: str, quiz_category: str, size: int):
    # click.echo("2c2d called")
    tctd = TwoComplementToDecimal()
    tctd.bit_number = 7
    output = "data/complemento_a_due_a_decimale.xml"
    generate_quizzes(tctd, category=quiz_category, file_name=file_name, output_dir=output_dir, default_outpath=output,
                     size=size)


@main.command("decimalec2", short_help="Da decimale a complemento a 2",
              help="Comando per la generazione di quiz del tipo "
                   "da decimale a complemento a 2")
@click.option('-f', '--file_name', default=None, show_default=True, prompt="Inserire il nome del file di destinazione",
              prompt_required=False)
@click.option('-d', '--output_dir', default=None, show_default=True,
              prompt="Inserire il nome della directory di destinazione", prompt_required=False)
@click.option('-c', '--quiz_category', default="Da decimale a complemento a 2", show_default=True)
@click.option('-s', '--size', type=int, default=50, help="Numero di domande generato", show_default=True,
              prompt="Inserire il numero di domande da generare")
def decimal_to_two_complement(file_name: str, output_dir: str, quiz_category: str, size: int):
    # click.echo("d22c called")
    tdtc = DecimalToTwoComplement()
    tdtc.bit_number = 7
    output = "data/decimale_a_complemento_a_due.xml"
    generate_quizzes(tdtc, category=quiz_category, file_name=file_name, output_dir=output_dir, default_outpath=output,
                     size=size)


@main.command("binariodecimale", short_help="Da binario a decimale",
              help="Comando per la generazione di quiz del tipo "
                   "da binario a decimale")
@click.option('-f', '--file_name', default=None, show_default=True, prompt="Inserire il nome del file di destinazione",
              prompt_required=False)
@click.option('-d', '--output_dir', default=None, show_default=True,
              prompt="Inserire il nome della directory di destinazione", prompt_required=False)
@click.option('-c', '--quiz_category', default="Da binario a decimale", show_default=True)
@click.option('-s', '--size', type=int, default=50, help="Numero di domande generato", show_default=True,
              prompt="Inserire il numero di domande da generare")
def binary_to_decimal(file_name: str, output_dir: str, quiz_category: str, size: int):
    # click.echo("2c2d called")
    tctd = BinaryToDecimal()
    tctd.bit_number = 5
    output = "data/binario_a_decimale.xml"
    generate_quizzes(tctd, category=quiz_category, file_name=file_name, output_dir=output_dir, default_outpath=output,
                     size=size)


@main.command("decimalebinario", short_help="Da decimale a binario",
              help="Comando per la generazione di quiz del tipo "
                   "da decimale a binario")
@click.option('-f', '--file_name', default=None, show_default=True, prompt="Inserire il nome del file di destinazione",
              prompt_required=False)
@click.option('-d', '--output_dir', default=None, show_default=True,
              prompt="Inserire il nome della directory di destinazione", prompt_required=False)
@click.option('-c', '--quiz_category', default="Da decimale a binario", show_default=True)
@click.option('-s', '--size', type=int, default=50, help="Numero di domande generato", show_default=True,
              prompt="Inserire il numero di domande da generare")
def decimal_to_binary(file_name: str, output_dir: str, quiz_category: str, size: int):
    # click.echo("d22c called")
    tdtc = DecimalToBinary()
    tdtc.bit_number = 5
    output = "data/decimale_a_binario.xml"
    generate_quizzes(tdtc, category=quiz_category, file_name=file_name, output_dir=output_dir, default_outpath=output,
                     size=size)


@main.command("funzioni", short_help="Funzioni booleane", help="Comando per la generazione di quiz del tipo "
                                                               "conversione da tavola di verità a decimale")
@click.option('-f', '--file_name', default=None, show_default=True, prompt="Inserire il nome del file di destinazione",
              prompt_required=False)
@click.option('-d', '--output_dir', default=None, show_default=True,
              prompt="Inserire il nome della directory di destinazione", prompt_required=False)
@click.option('-c', '--quiz_category', default="Funzioni booleane", show_default=True)
@click.option('-s', '--size', type=int, default=50, help="Numero di domande generato", show_default=True,
              prompt="Inserire il numero di domande da generare")
def funzioni(file_name: str, output_dir: str, quiz_category: str, size: int):
    # click.echo("d22c called")
    bf = BooleanFunction()
    bf.n = 4
    output = "data/funzioni_booleane.xml"
    generate_quizzes(bf, category=quiz_category, file_name=file_name, output_dir=output_dir, default_outpath=output,
                     size=size)


def generate_quizzes(quiz_generator: mxqg.Question, category: str, file_name: str, output_dir: str,
                     default_outpath: str, size: int):
    if file_name is not None:
        if output_dir is not None:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            outpath = os.path.join(output_dir, file_name)
        else:
            outpath = file_name
    else:
        outpath = default_outpath
    tctd_quizzes = mxqg.generate(quiz_generator,
                                 category=category,
                                 size=size)
    tctd_quizzes.save(outpath)


if __name__ == '__main__':
    main()
