import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from utils import metrics
from utils.util import make_src_mask, make_trg_mask
from trainer.trainer import Trainer


class TranslateTrainer(Trainer):
    def __init__(self, model, optimizer, criterion, cfg, logger, data_loader, valid_data_loader=None, lr_scheduler=None):
        super().__init__(model=model, optimizer=optimizer, criterion=criterion, cfg=cfg, logger=logger)
        
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler
        self.do_validation = self.valid_data_loader is not None
        self.device = 'cuda:0' if cfg['cuda'] else 'cpu'
        self.log_step = cfg['log_step']


    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for idx, (src, trg) in enumerate(self.data_loader):
            src = src.to(self.device)
            trg = trg.to(self.device) 

            src_mask = make_src_mask(src, self.data_loader.src_vocab, self.device)
            trg_mask = make_trg_mask(trg[:,:-1], self.data_loader.trg_vocab, self.device) 

            self.optimizer.zero_grad()
            
            output = self.model(src, trg[:,:-1], src_mask, trg_mask) # output = [batch_size, target_len-1, target_vocab_size]
            # trg = <sos>, token1, token2, token3, ... 
            # output = token1, token2, token3, ..., <eos>

            output_dim = output.shape[-1]
            
            output = output.contiguous().view(-1, output_dim) # output = [batch size * target_len - 1, target_vocab_size]
            
            trg = trg[:,1:].contiguous().view(-1) # target = [batch_size * targey_len - 1]

            loss = self.criterion(output, trg)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()
            total_loss += loss.item()
            if idx % self.log_step == 0:
                self.logger.info('Train Epoch: {}, {}/{} ({:.0f}%), Loss: {:.6f}'.format(epoch, 
                            idx, 
                            len(self.data_loader), 
                            idx * 100 / len(self.data_loader), 
                            loss.item()
                            ))

        self.logger.info('Train Epoch: {}, total Loss: {:.6f}, mean Loss: {:.6f}'.format(
                epoch,
                total_loss, 
                total_loss / len(self.data_loader)
                ))
        
        if self.do_validation:
            self.logger.debug("start validation")
            val_loss = self._valid_epoch()
        self.logger.info('Train Epoch: {}, validation loss is : {:.3f}'.format(epoch, val_loss))
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return val_loss, None

    def _valid_epoch(self):
        self.model.eval()
        val_loss = 0
        pred = []
        labels = []
        with torch.no_grad():
            for idx, (src, trg) in enumerate(self.valid_data_loader):
                src = src.to(self.device)
                trg = trg.to(self.device) 

                src_mask = make_src_mask(src, self.valid_data_loader.src_vocab, self.device)
                trg_mask = make_trg_mask(trg[:,:-1], self.data_loader.trg_vocab, self.device)

                output = self.model(src, trg[:,:-1], src_mask, trg_mask)
                output = F.log_softmax(output, dim=-1)
                output_dim = output.shape[-1]
                # output = [batch size * target_len - 1, target_vocab_size]
                output = output.contiguous().view(-1, output_dim)
                trg = trg[:,1:].contiguous().view(-1)
                
                val_loss += self.criterion(output, trg)
                
        return val_loss / len(self.valid_data_loader)
