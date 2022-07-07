# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from tokenizers import BertWordPieceTokenizer
import glob
from tqdm import tqdm
import os
import sys
import argparse



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='ELECTRA Info')
    parser.add_argument('-data', '-d',
                        default = None,
                        type = str,
                        help = "Path to dataset directory for tokenizer trainin.")
    parser.add_argument('-output_dir',
                        default = None,
                        type = str,
                        help = "Directory for writing output files.")
    parser.add_argument("-vocab_size", '-v',
                        default = 30522,
                        type = int,
                        required = False,
                        help = "Size of tokenizer's vocab")
    parser.add_argument("-lower_case", '-lc',
                        default = True,
                        type = bool,
                        required = False,
                        help = "Size of tokenizer's vocab")
    
    args = parser.parse_args()
    data_dir = args.data
    vocab_size = args.vocab_size
    lowercase = args.lower_case
    output_dir = args.output_dir
    
    # Create ouptut direactory
    if not os.path.exists('output'):
        os.mkdir('output')
    output_path = os.path.join('output',output_dir)
    if os.path.exists(output_path):
        print('output directory already exists')
        sys.exit()
    
    
    # Initialize an empty BERT tokenizer
    tokenizer = BertWordPieceTokenizer(
      clean_text=False,
      handle_chinese_chars=False,
      strip_accents=False,
      lowercase=lowercase,
    )
    
    # prepare text files to train vocab on them
    # files = ['aochildes.txt']
    files = file_paths = glob.glob(data_dir+"/*.txt")
    
    # train BERT tokenizer
    tokenizer.train(
      files,
      vocab_size=vocab_size,
      min_frequency=2,
      show_progress=True,
      special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],
      limit_alphabet=1000,
      wordpieces_prefix="##"
    )
    
    
    
    # Create output directory    
    os.mkdir(output_path)
    
    # Save tokenizer
    # tokenizer.save('owt-subset-10000-vobab.txt', pretty=False)
    tokenizer_file = ''.join([output_dir,'-',str(vocab_size),'-tokenizer.json'])
    tokenizer_path = os.path.join(output_path,tokenizer_file)
    tokenizer.save(tokenizer_path, pretty=False)
    
    
    # save vocab.txt
    vocab = tokenizer.get_vocab()
    vocab_file = ''.join([output_dir,'-',str(vocab_size),'-vocab.txt'])
    vocab_path = os.path.join(output_path,vocab_file)
    vocab_list = sorted(vocab, key=vocab.get)
    
    with open(vocab_path, 'w') as f:
    # with open('aochildes-10000-vobab.txt', 'w') as f:
        for token in vocab_list:
            line = ''.join([token,'\n'])
            f.write(line)
    
    
    
    
    
    
    
    
    
    
    
    
    
    