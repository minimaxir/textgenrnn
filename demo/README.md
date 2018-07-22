# 如何用textgenrnn处理中文

## 1. 什么是textgenrnn?

textgenrnn是建立在Keras和TensorFlow之上的，可用于生成`字级别`和`词级别`文本。网络体系结构使用注意力加权来加速训练过程并提高质量，并允许调整大量超参数，如RNN模型大小、RNN层和双向RNN。对细节感兴趣的读者，可以在Github上或类似的介绍博客文章中阅读有关textgenrnn及其功能和体系结构的更多信息。

Github项目地址: [https://github.com/minimaxir/textgenrnn](https://github.com/minimaxir/textgenrnn)

介绍博客(英文原版): [Generating Text with RNNs in 4 Lines of Code](https://www.kdnuggets.com/2018/06/generating-text-rnn-4-lines-code.html)

介绍博客(中文简译): [仅用四行代码实现RNN文本生成模型](https://yq.aliyun.com/articles/602825?utm_content=m_1000002413)

该Github项目的README以及各介绍博客中，都是基于英文文本进行处理的，少有文章介绍如何将textgenrnn应用到中文文本，所以我fork了原项目并添加了中文数据的训练与测试Demo，详见: [https://github.com/cheesezh/textgenrnn](https://github.com/cheesezh/textgenrnn)。

## 2. 如何用textgenrnn处理中文?

### 2.1. 准备中文数据

本文采用2600首与`春`相关的五言诗作为训练数据。部分数据示例：


    春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。
    慈母手中线，游子身上衣。谁言寸草心，报得三春晖。
    好雨知时节，当春乃发生。晓看红湿处，花重锦官城。
    红豆生南国，春来发几枝。愿君多采撷，此物最相思。
    国破山河在，城春草木深。白头搔更短，浑欲不胜簪。

可以在我的Github上边下载本文实验数据：[与“春”相关的五言诗](https://github.com/cheesezh/textgenrnn/blob/master/datasets/cn/5_chars_poem_2600.txt)，[与“春”相关的七言诗](https://github.com/cheesezh/textgenrnn/blob/master/datasets/cn/7_chars_poem_4200.txt)

### 2.2. 训练模型

```
from textgenrnn import textgenrnn
textgen = textgenrnn(name="my.poem")                   # 给模型起个名字，比如`my.poem`, 之后生成的模型文件都会以这个名字为前缀
textgen.reset()                                         # 重置模型
textgen.train_from_file(                                # 从数据文件训练模型
    file_path = '../datasets/cn/5_chars_poem_2600.txt',  # 文件路径
    new_model = True,                                   # 训练新模型
    num_epochs = 30,                                    # 训练轮数
    word_level = False,                                 # True:词级别，False:字级别
    rnn_bidirectional = True,                           # 是否使用Bi-LSTM
    max_length = 25,                                    # 一条数据的最大长度
)

```

还有其他的模型参数可以配置，主要包括以下几项：

```
config = {
        'rnn_layers': 2,
        'rnn_size': 128,
        'rnn_bidirectional': False,
        'max_length': 15,
        'max_words': 10000,
        'dim_embeddings': 100,
        'word_level': False,
        'single_text': False
    }
```

### 2.3. 生成数据

```
from textgenrnn import textgenrnn

textgen = textgenrnn(
  name="poem",
  weights_path='./poem_weights.hdf5',
  config_path='./poem_config.json',
  vocab_path='./poem_vocab.json'
)
textgen.generate(20, temperature=1.0)
```

生成数据样例:
```
傍海皆荒服，分符重汉臣。连年不见雪，到处即行春。

上喜老闲身，春来不得迷。章闲闲异意，亦随到情诗。

湖上花开尽，初度莫愁春。俗书百胜境，无哲老离秋。

三月小春来，闲人惊物华。且问听下意，多事翦彩光。

居春心在别，多是梅庭空。水梅人一梦，愁。场言诚微。成手。

南国无多雪，江春别离肠。非离菘楼畔，不殊问来花。

田家春事起，丁壮就东坡。予意在耕稼，因君问土宜。

春偏当春日，当桥如草心。如如如青我，当人有思生。

早春遥闻好，风晚景和。放诗此东方来，惟是日有情。

閒花淡心事，不作柳春枝。幸见散花里，何声满尔闻。

傍海皆荒服，分符重汉臣。连年不见雪，到处即行春。

子知千里，何年草旧居。遥知春还后日，何处二三难。
```

从生成的五言诗数据来看，大部分诗看上去还是比较正常的，但是也有一些不合理的数据，比如：

```
居春心在别，多是梅庭空。水梅人一梦，愁。场言诚微。成手。

子知千里，何年草旧居。遥知春还后日，何处二三难。

春偏当春日，当桥如草心。如如如青我，当人有思生。
```