import spacy
import re


class Tokenizer(object):
    def __init__(self, lang):
        spacy.prefer_gpu()
        self.nlp = spacy.load(lang)
        # self.punc = u"[^a-zA-Z0-9,\.\!\?\u4e00-\u9fa5，。！？]"  # 去除不常用标点
    
    def tokenizer(self, sentence):
        # sentence = re.sub(self.punc, ' ', sentence).lower()
        sentence = sentence.lower()
        token = [token.text.strip() for token in self.nlp.tokenizer(sentence) if token.text.strip() != ""]
        return token

if __name__ == "__main__":
    # zh : zh_core_web_md
    # en : en_core_web_md
    zh = 'zh_core_web_md'
    en = 'en_core_web_md'
    zh_tokenizer = Tokenizer(zh)
    en_tokenizer = Tokenizer(en)
    zh_text = "不幸的是，众议院共和党根本不想面对（关于银行的）社会科学的发现对科学（比如气候变化）的发现显然也是如此。"
    en_text = "Argentina's Diego Maradona, widely regarded as one of the world's greatest footballers, died of a heart attack on Wednesday, his lawyer said."

    zh_token = zh_tokenizer.tokenizer(zh_text)
    en_token = en_tokenizer.tokenizer(en_text)

    print(zh_token)
    print(en_token)