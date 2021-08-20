import os
import glob
import torch
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from torch.utils.data import Dataset, DataLoader, IterableDataset
import json
import pytorch_lightning as pl
import argparse
from transformers import T5Tokenizer, T5ForConditionalGeneration
from pprint import pprint

class AQUADataset(Dataset):
    def __init__(self, file, tok, max_len, pad_index = 0, ignore_index=-100):
        super().__init__()
        self.tok = tok
        self.max_len = max_len
        self.docs = self.read_file(file)
        self.len = len(self.docs)
        self.pad_index = pad_index
        self.ignore_index = ignore_index

    def read_file(self, file_path):
        data = []
        with open(file_path, 'r') as f:
            # it has 'correct', 'question', 'option', 'rationale'
            for line in f:
                data.append(json.loads(line))
        """ example:
{'correct': 'B',
 'input': 'Each child has 8 crayons and 15 apples . If there are 7 children , '
          'how many crayons are there in total ? <tab> A ) 22 <tab> B ) 56 '
          '<tab> C ) 12 <tab> D ) 36 <tab> E ) 10',
 'options': ['A ) 22', 'B ) 56', 'C ) 12', 'D ) 36', 'E ) 10'],
 'question': 'Each child has 8 crayons and 15 apples . If there are 7 children '
             ', how many crayons are there in total ?',
 'rationale': '8 * 7 = 56 . Answer is B .',
 'target': '8 * 7 = 56 . Answer is B . <tab> B'}
"""
        for d in data:
            d['input'] = f"{d['question']} <tab> {' <tab> '.join(d['options'])}"
            d['target'] = f"{d['rationale']} <tab> {d['correct']}"
        return data

    def add_padding_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.pad_index] *(self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]

        return inputs

    def add_ignored_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.ignore_index] *(self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]

        return inputs
    
    def __getitem__(self, idx):
        instance = self.docs[idx]
        input_ids = self.tok.encode(instance['input'])
        input_ids = self.add_padding_data(input_ids)

        label_ids = self.tok.encode(instance['target'])
        label_ids.append(self.tok.eos_token_id)
        dec_input_ids = [self.pad_index]
        dec_input_ids += label_ids[:-1]
        dec_input_ids = self.add_padding_data(dec_input_ids)
        label_ids = self.add_ignored_data(label_ids)

#         return (torch.tensor(input_ids),
#                 torch.tensor(dec_input_ids),
#                 torch.tensor(label_ids))
        return {'input_ids': np.array(input_ids, dtype=np.int_),
                'decoder_input_ids': np.array(dec_input_ids, dtype=np.int_),
                'labels': np.array(label_ids, dtype=np.int_)}
    
    def __len__(self):
        return self.len



class MathDataModule(pl.LightningDataModule):
    def __init__(self, train_file,
                 test_file, val_file, tok,
                 max_len=256,
                 batch_size=8,
                 num_workers=5):
        super().__init__()
        self.batch_size = batch_size
        self.max_len = max_len
        self.train_file_path = train_file
        self.val_file_path = val_file
        self.test_file_path = test_file
        if tok is None:
            self.tok = T5Tokenizer.from_pretrained("t5-base")
        else:
            self.tok = tok
        self.num_workers = num_workers

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--num_workers',
                            type=int,
                            default=5,
                            help='num of worker for dataloader')
        return parser

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage):
        # split dataset
        self.train = AQUADataset(self.train_file_path,
                                 self.tok,
                                 self.max_len)
        self.test = AQUADataset(self.test_file_path,
                                self.tok,
                                self.max_len)
        self.val = AQUADataset(self.val_file_path,
                                self.tok,
                                self.max_len)

    def train_dataloader(self):
        train = DataLoader(self.train,
                           batch_size=self.batch_size,
                           num_workers=self.num_workers, shuffle=True)
        return train

    def val_dataloader(self):
        val = DataLoader(self.val,
                         batch_size=self.batch_size,
                         num_workers=self.num_workers, shuffle=False)
        return val

    def test_dataloader(self):
        test = DataLoader(self.test,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False)
        return test


