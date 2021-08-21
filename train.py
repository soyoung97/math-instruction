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
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup

parser = argparse.ArgumentParser(description='Math instruction')

parser.add_argument('--checkpoint_path',
                    type=str,
                    help='checkpoint path')

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class ArgsBase():
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--train_file',
                            type=str,
                            default='AQuA/train.tok.json',
                            help='train file')

        parser.add_argument('--test_file',
                            type=str,
                            default='AQUA/test.tok.json',
                            help='test file')

        parser.add_argument('--val_file',
                            type=str,
                            default='AQUA/dev.tok.json',
                            help='val file')

        parser.add_argument('--batch_size',
                            type=int,
                            default=8,
                            help='')

        parser.add_argument('--max_len',
                            type=int,
                            default=512,
                            help='max seq len')
        return parser

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
                            default=14,
                            help='batch size for training (default: 96)')

        parser.add_argument('--lr',
                            type=float,
                            default=3e-5,
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

    def validation_step(self, batch, batch_idx):
        outs = self(batch)
        results = []
        loss = outs['loss']
        generated_outputs = self.model.generate(batch['input_ids'])
        correct = 0
        for i, gen in enumerate(generated_outputs):
            out = self.tokenizer.decode(gen)
            out = out.split('<extra_id_0>')
            if len(out) != 1: # if 1, it means zero occurances of extra_id_0 : treat it as wrong
                out = out[-1].replace('</s>', '').strip().upper()
                answer = self.int2ans[batch['answer'][i].item()]
                if answer == out:
                    correct += 1
                print(f"DEBUG: answer: {answer}, output: {out}")
        return (loss, correct)

    def validation_epoch_end(self, outputs):
        losses = []
        total_correct = 0
        for loss, correct in outputs:
            losses.append(loss)
            total_correct += correct
        print(f"Correct: {total_correct}, Total: {len(outputs)}")
        self.log('accuracy', total_correct/len(outputs))
        self.log('val_loss', torch.stack(losses).mean(), prog_bar=True)

if __name__ == '__main__':
    parser = Base.add_model_specific_args(parser)
    parser = ArgsBase.add_model_specific_args(parser)
    parser = MathDataModule.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    args.default_root_dir = 'logs'
    args.max_epochs = 100
    logging.info(args)

    model = T5ConditionalGeneration(args)

    dm = MathDataModule(args.train_file,
                        args.test_file,
                        args.val_file,
                        None,
                        batch_size=args.batch_size,
                        max_len=args.max_len,
                        num_workers=args.num_workers)
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='accuracy',
                                                       dirpath=args.default_root_dir,
                                                       filename='model_chp/{epoch:02d}-{val_loss:.3f}',
                                                       verbose=True,
                                                       save_last=True,
                                                       mode='min',
                                                       save_top_k=1)
    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(args.default_root_dir, 'tb_logs'))
    lr_logger = pl.callbacks.LearningRateMonitor()
    trainer = pl.Trainer.from_argparse_args(args, logger=tb_logger,
                                            callbacks=[checkpoint_callback, lr_logger])
    trainer.fit(model, dm)
