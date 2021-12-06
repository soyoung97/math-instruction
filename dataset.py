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
    def __init__(self, file, tok, max_len, mode,  pad_index = 0, ignore_index=-100):
        super().__init__()
        self.tok = tok
        self.max_len = max_len
        self.special_tok = '<extra_id_0>'
        self.mode = mode
        self.docs = self.read_file(file)
        self.filename = file
        self.len = len(self.docs)
        self.pad_index = pad_index
        self.ignore_index = ignore_index
        self.answer2int = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5}

    def read_file(self, file_path):
        data = []
        with open(file_path, 'r') as f:
            # it has 'correct', 'question', 'option', 'rationale'
            for line in f:
                data.append(json.loads(line))
        """ example:
{'correct': 'B',
 'input': 'Each child has 8 crayons and 15 apples . If there are 7 children , '
          'how many crayons are there in total ? {self.special_tok} A ) 22 {self.special_tok} B ) 56 '
          '{self.special_tok} C ) 12 {self.special_tok} D ) 36 {self.special_tok} E ) 10',
 'options': ['A ) 22', 'B ) 56', 'C ) 12', 'D ) 36', 'E ) 10'],
 'question': 'Each child has 8 crayons and 15 apples . If there are 7 children '
             ', how many crayons are there in total ?',
 'rationale': '8 * 7 = 56 . Answer is B .',
 'target': '8 * 7 = 56 . Answer is B . {self.special_tok} B'}
"""
        for d in data:
            if self.mode == 'normal':
                d['input'] = f"{d['question']} {self.special_tok} {f' {self.special_tok} '.join(d['options'])}"
                d['target'] = f"{d['correct']}"
            elif self.mode == 'explain':
                d['input'] = f"{d['question']} {self.special_tok} {f' {self.special_tok} '.join(d['options'])}"
                d['target'] = f"{d['rationale']} {self.special_tok} {d['correct']}"
            else:
                raise Exception(f"Mode not implemented: {self.mode}")
        return data

    def get_overfull_count(self):
        overfull_cnt = 0
        tgt_cnt = 0
        tgt_length = 0
        print(f"Starting get_overfull_count for {self.filename}...")
        for instance in tqdm(self.docs):
            input_ids = self.tok.encode(instance['input'])
            target_ids = self.tok.encode(instance['target'])
            if len(input_ids) >= self.max_len:
                overfull_cnt += 1
            if len(target_ids) >= self.max_len:
                tgt_cnt += 1
            tgt_length += len(target_ids)

        average_tgt_length = tgt_length/len(self.docs)
        print(f"counting done for {self.filename}")
        return overfull_cnt, tgt_cnt, len(self.docs), average_tgt_length

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
        #label_ids.append(self.tok.eos_token_id)
        #dec_input_ids = [self.pad_index]
        #dec_input_ids += label_ids[:-1]
        #dec_input_ids = self.add_padding_data(dec_input_ids)
        label_ids = self.add_ignored_data(label_ids)
        return {'input_ids': np.array(input_ids, dtype=np.int_),
         #       'decoder_input_ids': np.array(dec_input_ids, dtype=np.int_),
                'labels': np.array(label_ids, dtype=np.int_),
                'answer': np.array(self.answer2int[instance['correct']], dtype=np.int_)}

    def __len__(self):
        return self.len



class MathDataModule(pl.LightningDataModule):
    def __init__(self, train_file,
                 test_file, val_file, tok, mode,
                 max_len=512,
                 batch_size=8,
                 num_workers=5):
        super().__init__()
        self.batch_size = batch_size
        self.max_len = max_len
        self.train_file_path = train_file
        self.val_file_path = val_file
        self.test_file_path = test_file
        self.mode = mode
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
                            default=4,
                            help='num of worker for dataloader')
        return parser

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage):
        # split dataset
        self.train = AQUADataset(self.train_file_path,
                                 self.tok,
                                 self.max_len, self.mode)
        self.test = AQUADataset(self.test_file_path,
                                self.tok,
                                self.max_len, self.mode)
        self.val = AQUADataset(self.val_file_path,
                                self.tok,
                                self.max_len, self.mode)

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


