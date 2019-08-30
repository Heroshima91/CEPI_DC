import subprocess
import numpy as np
import pandas as pd


def learn_bpe(text_file, vocab_size):
    directory = 'C:/Users/louis.falissard/PycharmProjects/NLP_classifier/'
    #subprocess.run('subword-nmt learn-bpe -s ' + str(vocab_size) + ' < test_txt.txt > bpe.tok')

    subprocess.run('subword-nmt learn-bpe -s ' + str(vocab_size) + ' < ' + directory + text_file + ' > ' + directory + 'bpe.tok')


def apply_bpe(text_file, token_file):
    dir = 'C:/Users/louis.falissard/PycharmProjects/NLP_classifier/'
    subprocess.run('subword-nmt apply-bpe -c ' + dir + token_file + ' < ' + dir + text_file + ' > ' + dir + text_file + '.bpe')


"""
input_file = open('test_txt.txt', 'r')
code_file = open('codes.bpe', 'w')

code_file = open('codes.bpe', 'r')
output_file = open('test.bpe', 'w')

for line in input_file:
    output_file.write(bpe.process_line(line))"""
