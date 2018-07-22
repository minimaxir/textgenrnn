from textgenrnn import textgenrnn
textgen = textgenrnn(name="poem")                       # 给模型起个名字，比如`poem`, 之后生成的模型文件都会以这个名字为前缀
textgen.reset()                                         # 重置模型
textgen.train_from_file(                                # 从数据文件训练模型
    file_path = './datasets/cn/5_chars_poem_2600.txt',  # 文件路径
    new_model = True,                                   # 训练新模型
    num_epochs = 30,                                    # 训练轮数
    word_level = False,                                 # True:词级别，False:字级别
    rnn_bidirectional = True,                           # 是否使用Bi-LSTM
    max_length = 25,                                    # 一条数据的最大长度
)
