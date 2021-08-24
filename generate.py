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
from transformers import set_seed
import time
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description='Math instruction')

    parser.add_argument('--seed', type=int, help='Setting seed for reproducibility', default=0)
    parser.add_argument('--mode', type=str, default='normal')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--path', type=str, help='Model checkpoint path', default='logs/normal-prev/last.ckpt')
    parser.add_argument('--output', help='output file path', type=str, default='outputs/normal-prev-last.txt')
    parser.add_argument('--datatype', type=str, default='test') # could be test or val
    args = parser.parse_args()
    return args

def load_model(model_path, args):
    print(f"Loading model: {model_path} (Usually takes about 85-100 seconds)")
    cache_path = f"cache/{model_path.replace('/', '--')}.pickle"
    if os.path.exists(cache_path):
        start = time.time()
        with open(cache_path, 'rb') as f:
            model = pickle.load(f)
        end = time.time()
        print(f"Loading model from cache done - took {end-start} seconds")
    else:
        start = time.time()
        model = T5ConditionalGeneration.load_from_checkpoint(model_path, hparams={'mode': args.mode})
        end = time.time()
        print(f"Loading model done - took {end-start} seconds")
        start = time.time()
        with open(cache_path, 'wb') as f:
            pickle.dump(model, f)
        end = time.time()
        print(f"Saving pickle model done to {cache_path} - took {end-start} seconds")
    return model

def get_overfull_count(dm):
    #train_cnt, train_tgt, train_tot = dm.train.get_overfull_count()
    #val_cnt, val_tgt, val_tot = dm.val.get_overfull_count()
    test_cnt, test_tgt, test_tot, avlength  = dm.test.get_overfull_count()

   # print(f"Train input: {100* train_cnt/train_tot}% overfull ({train_cnt} / {train_tot})")
   # print(f"Train target: {100* train_tgt/train_tot}% overfull ({train_tgt} / {train_tot})")
    print(f"Test input: {100 * test_cnt/test_tot}% overfull ({test_cnt} / {test_tot})")
    print(f"Test target: {100 * test_tgt/test_tot}% overfull ({test_tgt} / {test_tot})")
    print(f"Average length for target: {avlength}")
   # print(f"Val input: {100* val_cnt/val_tot}% overfull ({val_cnt} / {val_tot})")
   # print(f"Val target: {100* val_tgt/val_tot}% overfull ({val_tgt} / {val_tot})")

def constrained_decoding(batch_id, input_ids): # WIP
    global orig_tokens_set
    # if last token is <extra_id_0>, generate only ABCDE
    # if last token is <pad>
        # if <extra_id_0> already exist, generate anything

    allowed_tokens_list = torch.arange(100)
    import pdb; pdb.set_trace()
    return allowed_tokens_list

def main():
    args = parse_args()
    set_seed(args.seed)
    dm = MathDataModule('AQuA/train.tok.json', 'AQuA/test.tok.json', 'AQuA/dev.tok.json', None, args.mode,
            batch_size=args.batch_size, max_len=512, num_workers=4)
    
    dm.setup(None)
    #get_overfull_count(dm)
    T5model = load_model(args.path, args).cuda()
    #T5model.model.config.max_length=256
    T5model.model.eval()
    #global orig_tokens_set
    #orig_tokens_set = torch.arange(32100)
    res = []
    explain = f"Output generated from: {args.path}"
    int2ans = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
    first = True
    total = 0
    correct = 0
    partial = 0
    wrong_format = 0
    inner_wrong_format = 0
    #ansans = []
    #bad_word_ids = [[22354], [14125], [19794], [2], [3,18,3,18]]#, [3,5,3,5], [5,6], [3,5,3,6]]
    if args.datatype == 'test':
        dataloader = dm.test_dataloader()
    else:
        dataloader = dm.val_dataloader()
    for batch in tqdm(dataloader):
        generated_outputs = T5model.model.generate(
                batch['input_ids'].cuda(),
                max_length=128,
                repetition_penalty=2.0) #,
                #temperature=1.5,
                #early_stopping=True,
                #num_beams=5,
               # bad_words_ids=bad_word_ids)
                #prefix_allowed_tokens_fn=constrained_decoding)
        for i, gen in enumerate(generated_outputs):
            if args.mode == 'explain':
                out = dm.tok.decode(gen)
                #labels = torch.where(batch['labels'][i] == -100, 0, batch['labels'][i])
                #label_ans = dm.tok.decode(labels).replace('<pad>', '').strip()
                #ansans.append(label_ans)
                # for debug
                res.append(out)
                original_output = out.replace('<pad>', '')
                out = format_explain_output(out)
                if out == 'wrong format':
                    wrong_format += 1 #fixed but just for debugging
                    penalty = 2.0
                    while out == 'wrong format':
                        penalty += 0.5
                        print(original_output)
                        gen = T5model.model.generate(batch['input_ids'][i].unsqueeze(0).cuda(), max_length=128, repetition_penalty=penalty)#, bad_words_ids=bad_word_ids)
                        out = dm.tok.decode(gen[0])
                        res[-1] = out
                        original_output = out
                        out = format_explain_output(out)
                if out == 'inner wrong format':
                    inner_wrong_format += 1
                else:
                    answer = int2ans[batch['answer'][i].item()]
                    if answer == out:
                        correct += 1
                        partial += 1 # partial scores are useless for explain
                    else:
                        labels = torch.where(batch['labels'][i] == -100, 0, batch['labels'][i])
                        label_ans = dm.tok.decode(labels).replace('<pad>', '').strip()
                        print(f"\n\n-----\n>>>Ques  : {dm.tok.decode(batch['input_ids'][i], skip_special_tokens=True)}\n>>>Label : {label_ans}\n>>>Out   : {original_output}\n>>>Answer: {answer}")
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
    acc_string += f' wrong format: {wrong_format} ({100 * wrong_format/total}%)'
    acc_string += f', inner wrong format: {inner_wrong_format} ({100 * inner_wrong_format/total}%)'
    explain += ", " + acc_string
    res.append(explain)
    with open(args.output, 'w') as f:
        f.write('\n'.join(res).strip())
    #with open('outputs/answer-explain-test.txt', 'w') as f:
    #    f.write('\n'.join(ansans).strip())
    print(f"Writing to {args.output} done!")
    print(acc_string)

def format_explain_output(raw):
    out = raw.split('<extra_id_0>')
    if len(out) == 1:
        return 'wrong format'
    out = out[-1].replace('</s>', '').replace('<pad>', '').strip().upper()
    if out not in ['A','B','C','D','E']:
        return 'inner wrong format'
    return out

if __name__ == '__main__':
    main()
