import torch
import torch.nn as nn
from torch.nn import functional as F
import argparse
from config import Config, Logger
from model import Transformer
from utils.vocab import Vocab
from utils.tokenizer import Tokenizer
from utils.util import make_src_mask, make_trg_mask


def translate_sentence(sentence, model, device, zh_vocab, en_vocab, zh_tokenizer, max_len = 100):
    model.eval()
    tokens = zh_tokenizer.tokenizer(sentence)
    tokens = ['<sos>'] + tokens + ['<eos>']
    print(tokens)
    tokens = [zh_vocab.word2id[word] for word in tokens]

    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)
    src_mask = make_src_mask(src_tensor, zh_vocab, device)
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)
    trg = [en_vocab.word2id['<sos>']]
    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg).unsqueeze(0).to(device)
        trg_mask = make_trg_mask(trg_tensor, en_vocab, device)
        with torch.no_grad():
            output = model.decoder(trg_tensor, enc_src, src_mask, trg_mask)
            output = model.fc(output)

        pred_token = output.argmax(2)[:,-1].item()
        trg.append(pred_token)
        if pred_token == en_vocab.word2id['<eos>']:
            break
    
    trg_tokens = [en_vocab.id2word[idx] for idx in trg]
    return trg_tokens

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sent', type=str, required=True)
    args = parser.parse_args()

    src_vocab = Vocab()
    trg_vocab = Vocab()
    
    logger = Logger()
    cfg = Config(logger, args)
    cfg.load_config('config.json')
    src_vocab.load(cfg.config['src_vocab'])
    trg_vocab.load(cfg.config['trg_vocab'])

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    zh = 'zh_core_web_md'
    en = 'en_core_web_md'

    model = Transformer(src_vocab_size=src_vocab.vocab_size, target_vocab_size=trg_vocab.vocab_size, device=device, **cfg.config)
    checkpoint = torch.load(cfg.config['resume_path'])
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    

    zh_tokenizer = Tokenizer(zh)

    sent = args.sent
    print(sent)

    res = translate_sentence(sent, model, device, src_vocab, trg_vocab, zh_tokenizer)
    print(res)
