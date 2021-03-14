import torch
import numpy as np
from config import Logger
from abc import abstractmethod
from utils.util import ensure_dir, check_dir

class Trainer(object):
    def __init__(self, model, optimizer, criterion, cfg, logger):

        self.model = model
        self.cfg = cfg
        self.logger = logger
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = cfg['epochs']
        self.start_epoch = 1 
        self.save_epochs = cfg['save_epochs']
        self.early_stop = cfg['early_stop']
        self.patience = cfg['patience']
        self.var_loss_min = np.Inf
        self.save_dir = cfg['save_dir']
        self.best_score = None
        self.counter = 0 
    
        if self.cfg['resume']:
            self._resume_checkpoint(cfg['resume_path'])

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        epoch: Current epoch number
        """
        raise NotImplementedError
    
    def train(self):
        for epoch in range(self.start_epoch, self.epochs+1):
            val_loss, metrics_score= self._train_epoch(epoch)
            score = -val_loss
            if self.early_stop:                             # 保存最有模型， 如果patience个epoch后， score没有提升，停止训练
                if self.best_score is None:
                    self.best_score = score
                    self._save_checkpoint(epoch, save_best=True)
                elif score < self.best_score:
                    self.counter += 1
                    self.logger.debug('EarlyStopping counter:{} out of {}'.format(self.counter, self.patience))
                    if self.counter >= self.patience:
                        self.logger.debug('Training early stops')
                        break
                else:
                    self.best_score = score
                    self._save_checkpoint(epoch, save_best=True)
                    self.counter = 0
            elif epoch % save_epochs == 0:
                self._save_checkpoint(epoch, save_best=True)

    def _save_checkpoint(self, epoch, save_best=False):
        ensure_dir(self.save_dir)
        state = {
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.cfg,
            'best_score': self.best_score
        }
        if save_best: 
            filename = str(self.save_dir + '/model_best.pt')
            torch.save(state, filename)
            self.logger.debug('Saving current best: {}...'.format(filename))
        else:
            filename = str(self.save_dir + '/checkpoint_epoch_{}.pt'.format(epoch))
            torch.save(state, filename)
            self.logger.debug('Saving checkpoint: {} ...'.format(filename))
        
    def _resume_checkpoint(self, path): # 恢复模型训练
        check_dir(path)
        self.logger.debug('Loading checkpoint: {}...'.format(path))
        checkpoint = torch.load(path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_score = checkpoint['best_score'] if 'best_score' in checkpoint else None
        self.model.load_state_dict(checkpoint['state_dict'])
        if checkpoint['config']['optimizer'] != self.cfg['optimizer']:
            self.logger.debug("Optimizer type given in config file is different from that of checkpoint."
                              "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.debug("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))