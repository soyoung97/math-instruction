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
from transformers import T5Tokenizer, T5ForConditionalGeneration, set_seed
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from model import T5ConditionalGeneration
from model import Base
parser = argparse.ArgumentParser(description='Math instruction')

parser.add_argument('--checkpoint_path',
                    type=str,
                    help='checkpoint path')

parser.add_argument('--log_dir', type=str, default='logs')

parser.add_argument('--seed', type=int, help='Setting seed for reproducibility', default=0)

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

        parser.add_argument('--max_len',
                            type=int,
                            default=512,
                            help='max seq len')
        parser.add_argument('--mode',
                            type=str, default='normal', help='model training mode: e.g. normal, explain, ...')
        return parser

if __name__ == '__main__':
    parser = Base.add_model_specific_args(parser)
    parser = ArgsBase.add_model_specific_args(parser)
    parser = MathDataModule.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    args.default_root_dir = args.log_dir
    args.max_epochs = 50
    args.gpus = 1 # -1
    args.num_workers = 4
    args.accelerator='dp'
    set_seed(args.seed)
    logging.info(args)

    model = T5ConditionalGeneration(args)
    #model.config.max_length = 512
    #model.config.early_stopping = True
    dm = MathDataModule(args.train_file,
                        args.test_file,
                        args.val_file,
                        None,
                        args.mode,
                        batch_size=args.batch_size,
                        max_len=args.max_len,
                        num_workers=args.num_workers)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_acc',
                                                       dirpath=args.default_root_dir,
                                                       filename='model_chp/{epoch:02d}-{val_loss:.3f}',
                                                       verbose=True,
                                                       save_last=True,
                                                       mode='min',
                                                       save_top_k=1)
    #tb_logger = pl_loggers.TensorBoardLogger(os.path.join(args.default_root_dir, 'tb_logs'))
    lr_logger = pl.callbacks.LearningRateMonitor()
    wandb_logger = WandbLogger()
    trainer = pl.Trainer.from_argparse_args(args, logger=wandb_logger,
                                            callbacks=[checkpoint_callback, lr_logger])
    trainer.fit(model, dm)
