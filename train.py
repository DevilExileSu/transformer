import torch
import argparse
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from config import Config,Logger
from utils.util import get_optimizer
from utils.vocab import Vocab
from torch.optim.lr_scheduler import LambdaLR
from model import Transformer
from trainer import TranslateTrainer 
from data import Zh2EnDataLoader
import os


parser = argparse.ArgumentParser()

# dataset parameter
parser.add_argument('--src_train_data', type=str, default='dataset/train.zh.token')
parser.add_argument('--trg_train_data', type=str, default='dataset/train.en.token')
parser.add_argument('--src_valid_data', type=str, default='dataset/val.zh.token')
parser.add_argument('--trg_valid_data', type=str, default='dataset/val.en.token')
parser.add_argument('--src_vocab', type=str, default='dataset/zh_vocab.pkl')
parser.add_argument('--trg_vocab', type=str, default='dataset/en_vocab.pkl')
parser.add_argument('--shuffle', type=bool, default=True)
parser.add_argument('--batch_size', type=int, default=32)

# model parameter
parser.add_argument('--h_dim', type=int, default=256)
parser.add_argument('--enc_n_layers', type=int, default=3)
parser.add_argument('--dec_n_layers', type=int, default=3)
parser.add_argument('--enc_n_heads', type=int, default=8) 
parser.add_argument('--dec_n_heads', type=int, default=8) 
parser.add_argument('--enc_dropout', type=float, default=0.1)
parser.add_argument('--dec_dropout', type=float, default=0.1)
parser.add_argument('--enc_pf_dim', type=int, default=512)
parser.add_argument('--dec_pf_dim', type=int, default=512)

# Loss function and Optimizer parameter
parser.add_argument('--lr', type=float, default=0.5)
parser.add_argument('--optimizer', choices=['sgd', 'adam', 'adamax'], default='adam', help='Optimizer: sgd, adagrad, adam or adamax.')
parser.add_argument('--l2', type=float, default=0.0)

# train parameter
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--save_dir', type=str, default='./saved_models')
parser.add_argument('--save_epochs', type=int, default=5, help='Save model checkpoints every k epochs.')
parser.add_argument('--early_stop', type=bool, default=True)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--resume_path', type=str, default='./saved_models/model_best.pt')
parser.add_argument('--log_step', type=int, default=20)

# other
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--config_file', type=str, default='./config.json')
parser.add_argument('--seed', type=int, default=1234)

args = parser.parse_args()
logger = Logger()

cfg = Config(logger=logger, args=args)
cfg.print_config()
cfg.save_config(cfg.config['config_file'])

torch.manual_seed(cfg.config['seed'])
torch.cuda.manual_seed(cfg.config['seed'])
torch.backends.cudnn.enabled = False
np.random.seed(cfg.config['seed'])

# vocab
src_vocab = Vocab()
src_vocab.load(cfg.config['src_vocab'])
trg_vocab = Vocab()
trg_vocab.load(cfg.config['trg_vocab'])


# data_loader
train_data_loader =  Zh2EnDataLoader(cfg.config['src_train_data'], cfg.config['trg_train_data'],
                                src_vocab, trg_vocab, cfg.config['batch_size'], cfg.config['shuffle'], logger)

valid_data_loader = Zh2EnDataLoader(cfg.config['src_valid_data'], cfg.config['trg_valid_data'], src_vocab, 
                                trg_vocab, cfg.config['batch_size'], cfg.config['shuffle'], logger)

# model 
device = 'cuda:0' if cfg.config['cuda'] else 'cpu'
model = Transformer(src_vocab_size=src_vocab.vocab_size, target_vocab_size=trg_vocab.vocab_size, device=device, **cfg.config)
model.to(device)
logger.info(model)

# optimizer and criterion
param = [p for p in model.parameters() if p.requires_grad]
optimizer = get_optimizer(cfg.config['optimizer'], param, lr=cfg.config['lr'])
# lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: cfg.config['lr'] / (epoch + 1))
criterion = nn.CrossEntropyLoss(ignore_index=trg_vocab.word2id['<pad>'])

#trainer
trainer = TranslateTrainer(model=model, optimizer=optimizer, criterion=criterion, cfg=cfg.config, logger=logger, 
                        data_loader=train_data_loader, valid_data_loader=valid_data_loader, lr_scheduler=None)
torch.cuda.set_device(0)
trainer.train()

