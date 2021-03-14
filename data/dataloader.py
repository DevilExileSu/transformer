import torch
import numpy as np

class BaseDataLoader(object):
    """
    Nonuse torch.utils.data
    """
    def __init__(self, ):
        """
        Initialization data file path, batch data size, shuffle data
        Read data from data file
        Preprocess the data
        Spilt the data according to batch_size
        """
        pass
    def __ltrg__(self):
        """
        How many batch
        """
        pass
    def __getitem__(self, index):
        """
        Return batch_size data pairs
        """
        pass
    def __read_data(self,):
        pass
    def __preprocess_data(self,):
        pass


class Zh2EnDataLoader(BaseDataLoader):
    def __init__(self, src_filename, trg_filename, src_vocab, trg_vocab, batch_size, shuffle, logger):
        super().__init__()
        self.src_filename = src_filename
        self.trg_filename = trg_filename
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.batch_size = batch_size 
        self.shuffle = shuffle
        self.logger = logger
        self.src_lines, self.trg_lines = self.__read_data()

    def __len__(self):
        return len(self.src_lines)
        
    def __getitem__(self, index):
        src_data = self.src_lines[index]
        trg_data = self.trg_lines[index]
     
        max_src_len = 0
        max_trg_len = 0
        src_batch_id = []
        trg_batch_id = []
        for src_tokens, trg_tokens in zip(src_data, trg_data):
            max_src_len = len(src_tokens) if len(src_tokens) > max_src_len else max_src_len
            max_trg_len = len(trg_tokens) if len(trg_tokens) > max_trg_len else max_trg_len
            src_batch_id.append([self.src_vocab.word2id[word] if word in self.src_vocab.word2id else self.src_vocab.word2id['<unk>'] for word in src_tokens])
            trg_batch_id.append([self.trg_vocab.word2id[word] if word in self.trg_vocab.word2id else self.trg_vocab.word2id['<unk>'] for word in trg_tokens])
    

        src = torch.LongTensor(self.batch_size, max_src_len).fill_(self.src_vocab.word2id['<pad>'])
        trg = torch.LongTensor(self.batch_size, max_trg_len).fill_(self.trg_vocab.word2id['<pad>'])
        for i in range(self.batch_size):
            src[i, :len(src_batch_id[i])] = torch.LongTensor(src_batch_id[i])
            trg[i, :len(trg_batch_id[i])] = torch.LongTensor(trg_batch_id[i])
        return src, trg

    def __read_data(self):
        self.logger.debug("-----------read data-----------")
        with open(self.src_filename, 'r', encoding='utf-8') as f:
            src_lines = np.array(f.readlines())
        with open(self.trg_filename, 'r', encoding='utf-8') as f: 
            trg_lines = np.array(f.readlines())
        assert len(src_lines) == len(trg_lines)
        if self.shuffle: 
            idx = np.random.permutation(len(src_lines))
            src_lines = src_lines[idx]
            trg_lines = trg_lines[idx]

        self.logger.debug("{} and {} has data {}".format(self.src_filename, self.trg_filename, len(src_lines)))
        return self.__preprocess_data(src_lines, trg_lines)

    def __preprocess_data(self, src_lines, trg_lines):
        self.logger.debug("-----------preprocess data-----------")
        src_lines = [['<sos>'] + line.strip().split('\t') + ['<eos>'] for line in src_lines]
        trg_lines = [['<sos>'] + line.strip().split('\t') + ['<eos>'] for line in trg_lines]

        src_lines = [src_lines[i:i+self.batch_size] for i in range(0, len(src_lines), self.batch_size)]
        trg_lines = [trg_lines[i:i+self.batch_size] for i in range(0, len(trg_lines), self.batch_size)]
        
        return src_lines, trg_lines