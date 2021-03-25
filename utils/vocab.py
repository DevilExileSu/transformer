import pickle
from tqdm import tqdm
from collections import Counter
from utils.tokenizer import Tokenizer
import multiprocessing

class Vocab(object):
    def __init__(self, min_freq=10):
        self.vocab = Counter()
        self.min_freq = min_freq 
        self.word2id = None
        self.id2word = None
        self.vocab_size = None
    
    def load(self, vocab_path):
        with open(vocab_path, 'rb') as f:
            tmp = pickle.load(f)
        self.vocab = tmp['vocab']
        self.min_freq = tmp['min_freq']
        self.word2id = tmp['word2id']
        self.id2word = tmp['id2word']
        self.vocab_size = tmp['vocab_size']

    def save(self, vocab_path):
        vocab = {
            "vocab" : self.vocab,
            "min_freq" : self.min_freq,
            "word2id" : self.word2id,
            "id2word" : self.id2word,
            "vocab_size" : self.vocab_size
        }
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab, f)

    def create(self, file_name, lang):
        nlp = Tokenizer(lang)

        print("-----------loading-----------")
        with open(file_name, encoding='utf-8') as f:
            lines = f.readlines()
        f = open(file_name+'.token', 'w', encoding='utf-8')
        for line in tqdm(lines):
            token = nlp.tokenizer(line)
            self.vocab.update(token)
            l = '\t'.join(token)
            f.write(l + '\n')
        
        tmp = self.vocab.most_common()
        tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
        tokens += [i[0] for i in tmp if i[1] > self.min_freq]
        self.word2id = {word:idx for idx, word in enumerate(tokens)}
        self.id2word = {idx:word for word, idx in self.word2id.items()}
        self.vocab_size = len(self.word2id)


if __name__ == "__main__":
    zh = 'zh_core_web_md'
    en = 'en_core_web_md'
    zh_train = '../dataset/train.zh'
    en_train = '../dataset/train.en'

    zh_valid = '../dataset/valid.zh'
    en_valid = '../dataset/valid.en'

    zh_test = '../dataset/test.zh'
    en_test = '../dataset/test.en'

    zh_vocab = Vocab()
    # en_nlp = Tokenizer(en)
    en_vocab = Vocab()
   
    zh_vocab.create(zh_train, zh)
    print(zh_vocab.vocab_size)
    zh_vocab.create(zh_valid, zh)
    print(zh_vocab.vocab_size)
    zh_vocab.create(zh_test, zh)
    print(zh_vocab.vocab_size)

    en_vocab.create(en_train, en)
    print(en_vocab.vocab_size)
    en_vocab.create(en_valid, en)
    print(en_vocab.vocab_size)
    en_vocab.create(en_test, en)
    print(en_vocab.vocab_size)

    zh_vocab.save('../dataset/zh_vocab.pkl')
    en_vocab.save('../dataset/en_vocab.pkl')




