import argparse
import logging
import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset
from dataset import AQUADataset, MathDataModule
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup


class Base(pl.LightningModule):
    def __init__(self, hparams, **kwargs) -> None:
        super(Base, self).__init__()
        self.save_hyperparameters(hparams)

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)

        parser.add_argument('--batch-size',
                            type=int,
                            default=8,
                            help='batch size for training (default: 96)')

        parser.add_argument('--lr',
                            type=float,
                            default=2e-5,
                            help='The initial learning rate')

        parser.add_argument('--warmup_ratio',
                            type=float,
                            default=0.1,
                            help='warmup ratio')

        return parser

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.lr, correct_bias=False)
        num_workers = self.hparams.num_workers
        data_len = len(self.train_dataloader().dataset)
        logging.info(f'number of workers {num_workers}, data length {data_len}')
        num_train_steps = int(data_len / (self.hparams.batch_size * num_workers) * self.hparams.max_epochs)
        logging.info(f'num_train_steps : {num_train_steps}')
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        logging.info(f'num_warmup_steps : {num_warmup_steps}')
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler, 
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]


class T5ConditionalGeneration(Base):
    def __init__(self, hparams, **kwargs):
        super(T5ConditionalGeneration, self).__init__(hparams, **kwargs)
        self.model = T5ForConditionalGeneration.from_pretrained("t5-base")
        self.model.train()
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.pad_token_id = 0
        self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
        self.int2ans = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
        try: # when we initialize model
            self.mode = hparams.mode
        except: # when we load checkpoint model
            self.mode = hparams['mode']
        self.init_val = True

    def forward(self, inputs):

        attention_mask = inputs['input_ids'].ne(self.pad_token_id).float()
        decoder_attention_mask = inputs['decoder_input_ids'].ne(self.pad_token_id).float()
        return self.model(input_ids=inputs['input_ids'],
                          attention_mask=attention_mask,
                          decoder_input_ids=inputs['decoder_input_ids'],
                          decoder_attention_mask=decoder_attention_mask,
                          labels=inputs['labels'], return_dict=True)


    def training_step(self, batch, batch_idx):
        outs = self(batch)
        loss = outs.loss
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def format_explain_output(self, raw):
        out = raw.split('<extra_id_0>')
        if len(out) == 1:
            return 'wrong format'
        out = out[-1].replace("</s>", '').replace("<pad>", '').strip().upper()
        return out

    def validation_step(self, batch, batch_idx):
        outs = self(batch)
        results = []
        loss = outs['loss']
        # add another loss for answer
        generated_outputs = self.model.generate(batch['input_ids'], max_length=128, repetition_penalty=2.0)
        correct = 0
        partial_correct = 0
        penalty = 2.0
        import pdb; pdb.set_trace()
        for i, gen in enumerate(generated_outputs):
            if self.mode == 'explain':
                out = self.tokenizer.decode(gen)
                out = self.format_explain_output(out)
                while penalty < 5 and out == 'wrong format':
                    penalty += 0.5
                    gen = self.model.generate(batch['input_ids'][i].unsqueeze(0), max_length=128, repetition_penalty=penalty)
                    out = self.tokenizer.decode(gen[0])
                    out = self.format_explain_output(out)
                answer = self.int2ans[batch['answer'][i].item()]
                if answer == out:
                    correct += 1
                if self.init_val:
                    print(f"EXPLAIN: answer: {answer}, output: {out}, same: {answer == out}")
            elif self.mode == 'normal':
                out = self.tokenizer.decode(gen, skip_special_tokens=True).upper().strip()
                answer = self.int2ans[batch['answer'][i].item()]
                if len(out) != 0:
                    if answer == out:
                        correct += 1
                    if answer == out[0]:
                        partial_correct += 1
                if self.init_val:
                    print(f"NORMAL: answer: {answer}, output: {out}, same: {answer == out}")
            else:
                raise Exception(f"Mode not implemented: {self.mode}")
        self.init_val = False
        val_acc, val_first_match_acc = correct/len(generated_outputs), partial_correct/len(generated_outputs)
        self.log("val_acc", val_acc, sync_dist=True)
        self.log("val_first_match_acc", val_first_match_acc, sync_dist=True)
        return (loss, val_acc, val_first_match_acc)

    def validation_epoch_end(self, outputs):
        losses, accs, fm_accs = [], [], []
        total_correct = 0
        for loss, acc, fm_acc in outputs:
            accs.append(acc)
            fm_accs.append(fm_acc)
            losses.append(loss)
        print(f"\n Validation Epoch END\nTotal: {len(outputs)}\nlosses:{losses}\naccs: {accs}\nfm_accs: {fm_accs}")
        self.log('epoch_acc', torch.stack(accs).mean(), sync_dist=True)
        self.log('epoch_first_match_acc', torch.stack(fm_accs).mean(), sync_dist=True)
        self.log('val_loss', torch.stack(losses).mean(), prog_bar=True, sync_dist=True)
        self.init_val = True

