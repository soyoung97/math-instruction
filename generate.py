import argparse
import logging
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader, Dataset
from dataset import AQUADataset, MathDataModule
from transformers import T5Tokenizer, T5ForConditionalGeneration, set_seed
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from model import T5ConditionalGeneration
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Math instruction')

    parser.add_argument('--seed', type=int, help='Setting seed for reproducibility', default=0)
    parser.add_argument('--mode', type=str, default='normal')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--path', type=str, help='Model checkpoint path', default='logs/normal-prev/last.ckpt')
    parser.add_argument('--output', help='output file path', type=str, default='outputs/normal-prev-last.txt')
    args = parser.parse_args()
    return args

def load_model(model_path, args):
    print(f"Loading model: {model_path} (Usually takes about 85-100 seconds)")
    start = time.time()
    model = T5ConditionalGeneration.load_from_checkpoint(model_path, hparams={'mode': args.mode})
    end = time.time()
    print(f"Loading model done - took {end-start} seconds")
    return model

def main():
    args = parse_args()
    dm = MathDataModule('AQuA/train.tok.json', 'AQuA/test.tok.json', 'AQuA/dev.tok.json', None, args.mode,
            batch_size=args.batch_size, max_len=512, num_workers=4)
    dm.setup(None)
    #train_cnt, train_tot = dm.train.get_overfull_count()
    #val_cnt, val_tot = dm.val.get_overfull_count()
    #test_cnt, test_tot = dm.test.get_overfull_count()

    #print(f"Train: {100* train_cnt/train_tot}% overfull ({train_cnt} / {train_tot})")
    #print(f"Test: {100 * test_cnt/test_tot}% overfull ({test_cnt} / {test_tot})")
    #print(f"Val: {100* val_cnt/val_tot}% overfull ({val_cnt} / {val_tot})")

    # load model
    T5model = load_model(args.path, args).cuda()
    res = []
    res.append(f"Output generated from: {args.path}")
    int2ans = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
    first = True
    total = 0
    correct = 0
    partial = 0
    for batch in tqdm(dm.test_dataloader()):
        generated_outputs = T5model.model.generate(batch['input_ids'].cuda())
        for i, gen in enumerate(generated_outputs):
            if args.mode == 'explain':
                out = dm.tok.decode(gen)
                res.append(out)
                out = out.split('<extra_id_0>')
                if len(out) != 1: # if 1, it means zero occurances of extra_id_0 : treat it as wrong
                    out = out[-1].replace('</s>', '').replace('<pad>', '').strip().upper()
                    answer = int2ans[batch['answer'][i].item()]
                    if answer == out:
                        correct += 1
                    if first: # for debug purposes
                        print(f"EXPLAIN: answer: {answer}, output: {out}")
                        first = False
                    if len(out) != 0:
                        if answer == out[0]:
                            partial += 1
                else:
                    out = ' '
            elif args.mode == 'normal':
                out = dm.tok.decode(gen, skip_special_tokens=True)
                res.append(out)
                out = out.upper()
                answer = int2ans[batch['answer'][i].item()]
                if answer == out:
                    correct += 1
                if first:
                    print(f"NORMAL: answer: {answer}, output: {out}")
                    first = False
                if len(out) > 0:
                    if answer == out[0]:
                        partial += 1
            else:
                raise Exception(f"Mode not implemented: {self.mode}")
            total += 1
    acc_string = f"accuracy: {correct}/{total} ({100 * correct / total} %), partial {partial}/{total} ({100 * partial / total} %)"
    res[0] = res[0] + ", " + acc_string
    with open(args.output, 'w') as f:
        f.write('\n'.join(res).strip())
    print(f"Writing to {args.output} done!")
    print(acc_string)
if __name__ == '__main__':
    main()
