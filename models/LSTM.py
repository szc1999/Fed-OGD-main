import os
 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import AG_NEWS
from torchtext.vocab import build_vocab_from_iterator
from tqdm import tqdm
 
class RNNnet(nn.Module):
    def __init__(self, len_vocab, embedding_size, hidden_size, num_class, num_layers, mode="lstm"):
        super(RNNnet, self).__init__()
        self.hidden = hidden_size
        self.num_layers = num_layers
        self.mode = mode
        self.embedding = nn.Embedding(len_vocab, embedding_size)
        if mode == "rnn":
            self.rnn = nn.RNN(embedding_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif mode == "lstm":
            self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif mode == "gru":
            self.rnn = nn.GRU(embedding_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_class)
 
    def forward(self, text):
        """
        :param text: [sentence_len, batch_size]
        :return:
        """
        # embedded:[sentence_len, batch_size, embedding_size]
        embedded = self.embedding(text)
        # output:[sentence_len, batch_size, hidden_size]
        # hidden:[1, batch_size, hidden_size]
        if self.mode == "rnn":
            output, hidden = self.rnn(embedded)
        elif self.mode == "lstm":
            output, (hidden, cell) = self.rnn(embedded)
        elif self.mode == "gru":
            output, hidden = self.rnn(embedded)
 
        return self.fc(hidden[-1])
    
class TextCNN(nn.Module):
    def __init__(self, len_vocab, embedding_dim=100, n_filters=100, filter_sizes=[3,4,5], output_dim=4, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(len_vocab, embedding_dim)
        self.conv_0 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(filter_sizes[0], embedding_dim))
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(filter_sizes[1], embedding_dim))
        self.conv_2 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(filter_sizes[2], embedding_dim))
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)
        conved_0 = nn.functional.relu(self.conv_0(embedded).squeeze(3))
        conved_1 = nn.functional.relu(self.conv_1(embedded).squeeze(3))
        conved_2 = nn.functional.relu(self.conv_2(embedded).squeeze(3))
        pooled_0 = nn.functional.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
        pooled_1 = nn.functional.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = nn.functional.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)
        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim=1))
        return self.fc(cat)
 
def loaddata(config):
    # Step1 加载数据集
    #######################################################################
 
    print("Step1: Loading DateSet")
    #######################################################################
    # 【数据集介绍】AG_NEWS, 新闻语料库，仅仅使用了标题和描述字段，
    # 包含4个大类新闻:World、Sports、Business、Sci/Tec。
    # 【样本数据】 120000条训练样本集（train.csv)， 7600测试样本数据集(test.csv);
    # 每个类别分别拥有 30,000 个训练样本及 1900 个测试样本。
    os.makedirs(config.datapath, exist_ok=True)
    train_dataset_o, test_dataset_o = AG_NEWS(root=config.datapath, split=("train", "test"))
    classes = ['World', 'Sports', 'Business', 'Sci/Tech']
 
    # for t in test_dataset_o:
    #     print(t)
    #     break
 
    return train_dataset_o, test_dataset_o, classes
 
 
def bulvocab(traindata):
    # Step2 分词，构建词汇表
    #######################################################################
    #
    print("Step2: Building VocabSet")
    #######################################################################
    tokenizer = get_tokenizer('basic_english')  # 基本的英文分词器，tokenizer会把句子进行分割，类似jieba
 
    def yield_tokens(data_iter):  # 分词生成器
        for _, text in data_iter:
            yield tokenizer(text)  # yield会构建一个类似列表可迭代的东西，但比起直接使用列表要少占用很多内存
 
    # 根据训练数据构建词汇表
    vocab = build_vocab_from_iterator(yield_tokens(traindata), specials=["<PAD>"])  # <unk>代指低频词或未在词表中的词
    # 词汇表会将token映射到词汇表中的索引上，注意词汇表的构建不需要用测试集
    vocab.set_default_index(vocab["<PAD>"])  # 设置默认索引，当某个单词不在词汇表vocab时（OOV)，返回该单词索引
    print(f"len vocab:{len(vocab)}")
    len_vocab = len(vocab)
 
    return vocab, len_vocab
 
 
def tensor_padding(tensor_list, seq_len):
    # 填充前两个张量
    padded_tensors = []
    for tensor in tensor_list:
        padding = (0, seq_len - len(tensor))  # 在末尾填充0
        padded_tensor = torch.nn.functional.pad(tensor, padding, mode='constant', value=0)
        padded_tensors.append(padded_tensor)
    return padded_tensors
 
 
def dateset2loader(config, vocab, traindata, testdata):
    tokenizer = get_tokenizer('basic_english')  # 基本的英文分词器，tokenizer会把句子进行分割，类似jieba
    # Step3 构建数据加载器 dataloader
    ##########################################################################
 
    print("Step3: DateSet -> Dataloader")
    ##########################################################################
    # text_pipeline将一个文本字符串转换为整数List, List中每项对应词汇表voca中的单词的索引号
    text_pipeline = lambda x: vocab(tokenizer(x))
 
    # label_pipeline将label转换为整数
    label_pipeline = lambda x: int(x) - 1
 
    # 加载数据集合，转换为张量
    def collate_batch(batch):
        """
        (3, "Wall") -> (2, "467")
        :param batch:
        :return:
        """
        label_list, text_list = [], []
        for (_label, _text) in batch:
            label_list.append(label_pipeline(_label))
            processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
 
        # 指定句子长度统一的标准
        # if config.seq_mode == "min":
        #     seq_len = min(len(item) for item in text_list)
        # elif config.seq_mode == "max":
        #     seq_len = max(len(item) for item in text_list)
        # elif config.seq_mode == "avg":
        seq_len = sum(len(item) for item in text_list) / len(text_list)
        # elif isinstance(config.seq_mode, int):
        #     seq_len = config.seq_mode
        # else:
        #     seq_len = min(len(item) for item in text_list)
        seq_len = int(seq_len)
        # 每一个batch里统一长度
        batch_seq = torch.stack(tensor_padding(text_list, seq_len))
 
        label_list = torch.tensor(label_list, dtype=torch.int64)
        return batch_seq, label_list
 
    train_dataloader = DataLoader(traindata, batch_size=config.batchsize, shuffle=True, collate_fn=collate_batch)
    test_dataloader = DataLoader(testdata, batch_size=config.batchsize, shuffle=True, collate_fn=collate_batch)
 
    return train_dataloader, test_dataloader
 
 
def model_train(config, len_vocab, classes, train_dataloader, test_dataloader):
    # 构建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rnn_model = RNNnet(
        len_vocab=len_vocab,
        embedding_size=config.embedding_size,
        hidden_size=config.hidden_size,
        num_class=len(classes),
        num_layers=config.num_layers,
        mode=config.mode
    )
 
    optimizer = torch.optim.Adam(rnn_model.parameters(), lr=config.l_r)
    loss_fn = nn.CrossEntropyLoss()
    rnn_model.train()
    rnn_model.to(device)
    # 训练模型
    LOSS = []
    ACC = []
    TACC = []
    os.makedirs(config.savepath, exist_ok=True)
    best_acc = 0
    for epoch in range(config.epochs):
        loop = tqdm(train_dataloader, desc='Train')
        total_loss, total_acc, count, i = 0, 0, 0, 0
        rnn_model.train()
        for idx, (text, label) in enumerate(loop):
            text = text.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = rnn_model(text)  # 预测
 
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()
            y_delta = rnn_model.state_dict()
            predict = torch.argmax(output, dim=1)  # 判断与原标签是否一样
            acc = (predict == label).sum()
            total_loss += loss.item()
            total_acc += acc.item()
            count += len(label)
            i += 1
            # 打印过程
            loop.set_description(f'Epoch [{epoch + 1}/{config.epochs}]')
            loop.set_postfix(loss=round(loss.item(), 4), acc=(round(acc.item() / len(label), 4) * 100))
        print(
            f"epoch_loss:{round(total_loss / i, 4)}\nepoch_acc:{round(total_acc / count, 4) * 100}%")
        # 保存模型参数
 
        LOSS.append(round(total_loss / i, 4))
        ACC.append(round((total_acc / count) * 100, 4))
 
        rnn_model.eval()
        test_loop = tqdm(test_dataloader)
        total_acc, count = 0, 0
        for idx, (text, label) in enumerate(test_loop):
            text = text.to(device)
            label = label.to(device)
            output = rnn_model(text)
            predict = torch.argmax(output, dim=1)  # 判断与原标签是否一样
            acc = (predict == label).sum()
            total_acc += acc.item()
            count += len(label)
        print(f"测试集精度：{round((total_acc / count) * 100, 2)}%")
        temp_acc = round((total_acc / count) * 100, 2)
        TACC.append(temp_acc)
        if temp_acc > best_acc:
            best_acc = temp_acc
            torch.save(rnn_model.state_dict(), f"{config.savepath}/{config.modelpath}")
 
    print(f"LOSS_array:{LOSS}")
    print(f"ACC_array:{ACC}")
    print(f"TACC_array:{TACC}")
    with open(config.logpath, 'w') as f:
        f.write(f"LOSS_array:{LOSS}\nACC_array:{ACC}\nTACC_array:{TACC}")
 
 
def modeltest(config, len_vocab, classes, test_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 测试模型
    rnn_model_test = RNNnet(
        len_vocab=len_vocab,
        embedding_size=config.embedding_size,
        hidden_size=config.hidden_size,
        num_class=len(classes),
        num_layers=config.num_layers,
        mode=config.mode
    )
    rnn_model_test.load_state_dict(torch.load(f"{config.savepath}/{config.modelpath}"))
    rnn_model_test.eval()
    rnn_model_test.to(device)
    test_loop = tqdm(test_dataloader)
    total_acc, count = 0, 0
    for idx, (text, label) in enumerate(test_loop):
        text = text.to(device)
        label = label.to(device)
        output = rnn_model_test(text)
        predict = torch.argmax(output, dim=1)  # 判断与原标签是否一样
        acc = (predict == label).sum()
        total_acc += acc.item()
        count += len(label)
    print(f"测试集精度：{round((total_acc / count) * 100, 2)}%")
 
 
def plot_result(logpath, mode):
    mode_set = ["loss", 'train_accuracy', 'test_accuracy']
    if mode not in mode_set:
        return "wrong mode"
    color = ['blue', 'red', 'black']
    with open(logpath, "r") as f:
        line = f.readlines()[mode_set.index(mode)]
        y = eval(line[line.index(':') + 1:])
        x = [i for i in range(len(y))]
 
        plt.figure()
 
        # 去除顶部和右边框框
        ax = plt.axes()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
 
        plt.xlabel('epoch')
        plt.ylabel(f'{mode}')
 
        plt.plot(x, y, color=color[mode_set.index(mode)], linestyle="solid", label=f"train {mode}")
        plt.legend()
 
        plt.title(f'train {mode} curve')
        plt.show()
        plt.savefig(f"{mode}.png")
 
 
class Config:
    def __init__(self):
        #######################################
        # 数据集使用的是AG_NEWS，实现文本分类任务
        #######################################
        self.datapath = '../data/AG_news'
        self.savepath = './save_model'
        self.modelpath = 'rnn_model.pt'
        self.logpath = 'log_best.txt'
        self.embedding_size = 32
        self.hidden_size = 64
        self.num_layers = 1  # rnn的层数
        self.l_r = 1e-3
        self.epochs = 50
        self.batchsize = 1024
        self.plotloss = True
        self.plotacc = True
        self.train = True
        self.test = True
        self.seq_mode = "avg"  # seq_mode:min、max、avg、也可输入一个数字自定义长度
        self.test_one = False
        self.test_self = "./test_self.txt"
        self.mode = "lstm"
 
    def parm(self):
        print(
            f"datapath={self.datapath}\n"
            f"savepath={self.savepath}\n"
            f"modelpath={self.modelpath}\n"
            f"logpath={self.logpath}\n"
            f"embedding_size={self.embedding_size}\n"
            f"hidden_size={self.hidden_size}\n"
            f"num_layers={self.num_layers}\n"
            f"l_r={self.l_r}\n"
            f"epochs={self.epochs}\n"
            f"batchsize={self.batchsize}\n"
            f"plotloss={self.plotloss}\n"
            f"plotacc={self.plotacc}\n"
            f"train={self.train}\n"
            f"test={self.test}\n"
            f"seq_mode={self.seq_mode}\n"
            f"test_one={self.test_one}\n"
            f"test_self={self.test_self}\n"
            f"mode={self.mode}\n"
        )
 
 
def simple(vocab, text_one):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 测试模型
    rnn_model_test = RNNnet(
        len_vocab=len_vocab,
        embedding_size=config.embedding_size,
        hidden_size=config.hidden_size,
        num_class=len(classes),
        num_layers=config.num_layers,
        mode=config.mode
    )
    rnn_model_test.load_state_dict(torch.load(f"{config.savepath}/{config.modelpath}"))
    rnn_model_test.eval()
    rnn_model_test.to(device)
 
    text_pipeline = lambda x: vocab(tokenizer(x))
    tokenizer = get_tokenizer('basic_english')  # 基本的英文分词器，tokenizer会把句子进行分割，类似jieba
 
    for text in text_one:
        text_one_tensor = torch.tensor(text_pipeline(text), dtype=torch.int64)
        text_one_tensor = text_one_tensor.to(device)
        print(f"预测标签为：{torch.argmax(rnn_model_test(text_one_tensor)).item() + 1}")
 
 
if __name__ == "__main__":
    config = Config()
    config.parm()
    if config.train:
        train_dataset_o, test_dataset_o, classes = loaddata(config)
        vocab, len_vocab = bulvocab(train_dataset_o)
        train_dataloader, test_dataloader = dateset2loader(config, vocab, train_dataset_o, test_dataset_o)
        model_train(config, len_vocab, classes, train_dataloader, test_dataloader)
    if config.test:
        train_dataset_o, test_dataset_o, classes = loaddata(config)
        vocab, len_vocab = bulvocab(train_dataset_o)
        train_dataloader, test_dataloader = dateset2loader(config, vocab, train_dataset_o, test_dataset_o)
        modeltest(config, len_vocab, classes, test_dataloader)
    if config.plotloss:
        plot_result(config.logpath, 'loss')
    if config.plotacc:
        plot_result(config.logpath, 'train_accuracy')
        plot_result(config.logpath, 'test_accuracy')
    if config.test_one:
        train_dataset_o, test_dataset_o, classes = loaddata(config)
        vocab, len_vocab = bulvocab(train_dataset_o)
        with open(config.test_self, "r") as f:
            simple(vocab, f.readlines())
