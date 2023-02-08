# Transformer

Transformer: seq2seq，机器翻译，中文--英文

基于[模板](https://github.com/DevilExileSu/pytorch-template)编写


### 预处理
首先使用Vocab类运行，生成字典和以'\t'为分隔符的分词数据集。

使用到了[spaCy](https://github.com/explosion/spaCy)，需要在[spaCy/releases](https://github.com/explosion/spaCy/releases)下载预训练模型。
```python
from utils.vocab import Vocab

zh = 'zh_core_web_md'
en = 'en_core_web_md'
zh_train = 'dataset/train.zh'
en_train = 'dataset/train.en'

zh_valid = 'dataset/valid.zh'
en_valid = 'dataset/valid.en'

zh_vocab = Vocab()
en_vocab = Vocab()

zh_vocab.create(zh_train, zh)
zh_vocab.create(zh_valid, zh)

en_vocab.create(en_train, en)
en_vocab.create(en_valid, en)

zh_vocab.save('dataset/zh_vocab.pkl')
en_vocab.save('dataset/en_vocab.pkl')
```

### 训练模型
```python 
python train.py --batch_size 32 --h_dim 256 --lr 0.0005 --epochs 50
```

这是 README 中测试使用到的模型参数，可以下载替换使用。

链接：https://pan.baidu.com/s/18D9UXTYeIYmOy3dfPVopZw 
提取码：sxho

### 翻译
```python 
python test.py --sent 这是一个例子
"""
2022-09-06 23:57:53,928 [DEBUG]: Config loaded from file config.json
这是一个例子
['<sos>', '这是', '一个', '例子', '<eos>']
['<sos>', 'this', 'is', 'a', 'example', 'of', 'this', 'example', '<eos>']
"""
```
