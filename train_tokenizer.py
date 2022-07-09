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
    parser.add_argument("-vocab_size", '-vs',
                        default = 30522,
                        type = int,
                        required = False,
                        help = "Size of tokenizer's vocab")
    parser.add_argument("-limit_alphabet", '-lv',
                        default = 1000,
                        type = int,
                        required = False,
                        help = "Size of tokenizer's limit alphabet")
    parser.add_argument("-lower_case", '-lc',
                        default = True,
                        type = bool,
                        required = False,
                        help = "Size of tokenizer's vocab")
    
    # Parse args
    args = parser.parse_args()
    data_dir = args.data
    vocab_size = args.vocab_size
    limit_alphabet = args.limit_alphabet
    lowercase = args.lower_case
    
    # Create ouptut direactory
    if not os.path.exists('output'):
        os.mkdir('output')
        
    output_dir = data_dir.rstrip('/')
    output_dir = output_dir.rstrip('\\')
    output_dir = os.path.split(output_dir)[1]
    output_dir = '-'.join([output_dir,'vs'+str(vocab_size),'la'+str(limit_alphabet)])
        
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
    files = file_paths = glob.glob(data_dir+"/*.txt")
    
    # train BERT tokenizer
    tokenizer.train(
      files,
      vocab_size=vocab_size,
      min_frequency=2,
      show_progress=True,
      special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],
      limit_alphabet=limit_alphabet,
      wordpieces_prefix="##"
    )
    
    # Create output directory    
    os.mkdir(output_path)
    
    # Save tokenizer
    tokenizer_file = '-'.join([output_dir,'tokenizer.json'])
    tokenizer_path = os.path.join(output_path,tokenizer_file)
    tokenizer.save(tokenizer_path, pretty=False)
    
    
    # save vocab.txt
    vocab = tokenizer.get_vocab()
    vocab_file = '-'.join([output_dir,'vocab.txt'])
    vocab_path = os.path.join(output_path,vocab_file)
    vocab_list = sorted(vocab, key=vocab.get)
    
    with open(vocab_path, 'w') as f:
        for token in vocab_list:
            line = ''.join([token,'\n'])
            f.write(line)
    
    
    
    
    
    
    
    
    
    
    
    
    
    