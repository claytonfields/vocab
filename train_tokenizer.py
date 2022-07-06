# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from tokenizers import BertWordPieceTokenizer
import glob
from tqdm import tqdm
import pickle
import os
import argparse



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='ELECTRA Info')
    # parser.add_argument('model_path',
    #                     default = None,
    #                     type = str,
    #                     help = "Path to the PyTorch checkpoint.")
    # parser.add_argument("-epochs", '-e',
    #                     default = 1,
    #                     type = str,
    #                     required = False,
    #                     help = "Number of training epochs")
    # args = parser.parse_args()
    
    
    # Initialize an empty BERT tokenizer
    tokenizer = BertWordPieceTokenizer(
      clean_text=False,
      handle_chinese_chars=False,
      strip_accents=False,
      lowercase=True,
    )
    
    # prepare text files to train vocab on them
    # files = ['aochildes.txt']
    files = file_paths = glob.glob("aochildes/*.txt")
    
    # train BERT tokenizer
    tokenizer.train(
      files,
      vocab_size=10000,
      min_frequency=2,
      show_progress=True,
      special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],
      limit_alphabet=1000,
      wordpieces_prefix="##"
    )
    
    # save the vocab
    # tokenizer.save('owt-subset-10000-vobab.txt', pretty=False)
    
    tokenizer.save('childes-10000-tokenizer.json', pretty=False)
    
    # with open('childes-10000-vocab.txt', 'w') as f:
    # # with open('aochildes-10000-vobab.txt', 'w') as f:
    #     for key in vocab.keys():
    #         line = ''.join([key,'\n'])
    #         f.write(line)