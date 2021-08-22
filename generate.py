import argparse
import logging
import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader, Dataset
from dataset import AQUADataset, MathDataModule
from transformers import T5Tokenizer, T5ForConditionalGeneration, set_seed
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup


def parse_args():
    parser = argparse.ArgumentParser(description='Math instruction')

    parser.add_argument('--seed', type=int, help='Setting seed for reproducibility', default=0)
    parser.add_argument('--mode', type=str, default='normal')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    dm = MathDataModule('AQuA/train.tok.json', 'AQuA/test.tok.json', 'AQuA/val.tok.json', None, args.mode,
            batch_size=args.batch_size, max_len=512, num_workers=4)
    train_cnt, train_tot = dm.train.get_overfull_count()
    val_cnt, val_tot = dm.val.get_overfull_count()
    test_cnt, test_tot = dm.test.get_overfull_count()

    print(f"Train: {100* train_cnt/train_tot}% overfull")
    print(f"Test: {100 * test_cnt/test_tot}% overfull")
    print(f"Val: {100* val_cnt/val_tot}% overfull")


    int2ans = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E'}

    # TODO: load model
    for batch in dm.test_dataloader():
        generated_outputs = self.model.generate(batch['input_ids'])
        correct = 0
        for i, gen in enumerate(generated_outputs):
            if args.mode == 'explain':
                out = dm.tokenizer.decode(gen)
                out = out.split('<extra_id_0>')
                if len(out) != 1: # if 1, it means zero occurances of extra_id_0 : treat it as wrong
                    out = out[-1].replace('</s>', '').strip().upper()
                    answer = int2ans[batch['answer'][i].item()]
                    if answer == out:
                        correct += 1
                    print(f"explain: answer: {answer}, output: {out}")
            elif self.mode == 'normal':
                out = self.tokenizer.decode(gen, skip_special_tokens=True).upper()
                answer = self.int2ans[batch['answer'][i].item()]
                if answer == out:
                    correct += 1
                print(f"NORMAL: answer: {answer}, output: {out}")
            else:
                raise Exception(f"Mode not implemented: {self.mode}")
        return (loss, correct)

if __name__ == '__main__':
    main()
