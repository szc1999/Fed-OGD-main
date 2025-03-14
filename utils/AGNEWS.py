import numpy as np
import os
import sys
import random
import torch
from torch.utils.data import Dataset
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
# from utils.dataset_utils import check, separate_data, split_data, save_file
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
import torch.nn as nn


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

def tensor_padding(tensor_list, seq_len):
    # 填充前两个张量
    padded_tensors = []
    for tensor in tensor_list:
        padding = (0, seq_len - len(tensor))  # 在末尾填充0
        padded_tensor = torch.nn.functional.pad(tensor, padding, mode='constant', value=0)
        padded_tensors.append(padded_tensor)
    return padded_tensors

def collate_batch(batch):
    # label_pipeline将label转换为整数
    # label_pipeline = lambda x: int(x) - 1
    label_list, text_list = [], []
    for (_text, _label) in batch:
        label_list.append(_label)
        processed_text = torch.tensor(_text, dtype=torch.int64)
        text_list.append(processed_text)

    # 指定句子长度统一的标准
    # if config.seq_mode == "min":
    #     seq_len = min(len(item) for item in text_list)
    # elif config.seq_mode == "max":
    # seq_len = max(len(item) for item in text_list)
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

def tokenizer(text):
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(
        map(tokenizer, iter(text)), 
        specials = ['<pad>'],
        # specials = ['<pad>', 'cls', '<unk>', '<eos>'],
        # special_first = True, 
        # max_tokens = max_tokens 
    )
    vocab.set_default_index(vocab['<pad>'])
    # vocab.set_default_index(vocab['<unk>'])
    text_pipeline = lambda x: vocab(tokenizer(x))

    text_list = []
    for t in text:
        tokens = text_pipeline(t)
        # tokens = [vocab['<cls>']] + text_pipeline(t)
        # if max_len>len(tokens):
        #     padding = [0 for i in range(max_len - len(tokens))]
        #     tokens.extend(padding)
        text_list.append(tokens)
    return vocab, text_list

class AGNewsDataset(Dataset):
    def __init__(self, root, train=True):
        # 初始化数据
        dataset, dataset2= AG_NEWS(root=root, split=('train', 'test'))
        # trainlabel, traintext = list(zip(*trainset))
        # testlabel, testtext = list(zip(*testset))
        
        dataset_text = []
        dataset_label = []

        for l, t in dataset:
            dataset_text.append(t)
            dataset_label.append(l)
        for l, t in dataset2:
            dataset_text.append(t)
            dataset_label.append(l)

        # 生成一个索引列表
        indices = list(range(len(dataset_label)))
        # 打乱索引列表的顺序
        random.shuffle(indices)
        # 根据打乱后的索引重新排列数据和标签列表
        if train :
            shuffled_text = [dataset_text[i] for i in indices]
            shuffled_label = [dataset_label[i] for i in indices]
        else:
            shuffled_text = [dataset_text[i] for i in indices][7600:]
            shuffled_label = [dataset_label[i] for i in indices][7600:]
        
        # if train == True:
        #     dataset_text.extend(traintext)
        #     dataset_label.extend(trainlabel)
        #     dataset_text.extend(testtext)
        #     dataset_label.extend(testlabel)
        # else:
        #     dataset_text.extend(testtext)
        #     dataset_label.extend(testlabel)

        num_classes = len(set(dataset_label))

        # self.vocab, text_list = tokenizer(dataset_text)
        self.vocab, text_list = tokenizer(shuffled_text)
        label_pipeline = lambda x: int(x) - 1
        # label_list = [label_pipeline(l) for l in dataset_label]
        label_list = [label_pipeline(l) for l in shuffled_label]

        text_lens = [len(text) for text in text_list]
        self.text_list = [torch.tensor(text, dtype=torch.int64) for text in text_list]
        self.targets = torch.tensor(label_list, dtype=torch.int64)
        
    def __len__(self):
        # 返回数据集的长度
        return len(self.targets)
    def __getitem__(self, idx):
        # 根据索引返回数据样本
        text, label = self.text_list[idx], self.targets[idx]
        return text, label




if __name__ == "__main__":
    # config = Config()
    # train_dataset_o, test_dataset_o, classes = loaddata(config)
    # vocab, len_vocab = bulvocab(train_dataset_o)
    # train_dataloader, test_dataloader = dateset2loader(config, vocab, train_dataset_o, test_dataset_o)
    
    agnews_dataset = AGNewsDataset(root="../data/AG_news",train=True)
    # text, label = agnews_dataset[15]
    # print(f"Label: {label}, Text: {text}")
    loader_train = DataLoader(agnews_dataset, batch_size=1024, shuffle=True, collate_fn=collate_batch)
    # loader_train = train_dataloader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rnn_model = RNNnet(
        len_vocab=len(agnews_dataset.vocab),
        embedding_size=32,
        hidden_size=64,
        num_class=4,
        num_layers=1,
        mode='rnn'
    )

    t_agnews_dataset = AGNewsDataset(root="../data/AG_news",train=False)
    loader_test = DataLoader(t_agnews_dataset, batch_size=1024, shuffle=True, collate_fn=collate_batch)
    # loader_test = test_dataloader
    optimizer = torch.optim.Adam(rnn_model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    rnn_model.train()
    rnn_model.to(device)
    # 训练模型
    LOSS = []
    ACC = []
    TACC = []
    best_acc = 0
    for epoch in range(50):
        loop = tqdm(loader_train, desc='Train')
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
            loop.set_description(f'Epoch [{epoch + 1}/{50}]')
            loop.set_postfix(loss=round(loss.item(), 4), acc=(round(acc.item() / len(label), 4) * 100))
        print(
            f"epoch_loss:{round(total_loss / i, 4)}\nepoch_acc:{round(total_acc / count, 4) * 100}%")
        # 保存模型参数
 
        LOSS.append(round(total_loss / i, 4))
        ACC.append(round((total_acc / count) * 100, 4))
 
        rnn_model.eval()
        test_loop = tqdm(loader_test)
        total_loss, total_acc, count, i = 0, 0, 0, 0
        for idx, (text, label) in enumerate(loop):
            text = text.to(device)
            label = label.to(device)
            output = rnn_model(text)
            predict = torch.argmax(output, dim=1)  # 判断与原标签是否一样
            acc = (predict == label).sum()
            total_acc += acc.item()
            count += len(label)
            loss = loss_fn(output, label)
            y_delta = rnn_model.state_dict()
            total_loss += loss.item()
            i+=1
        print(f"测试集精度：{round((total_acc / count) * 100, 2)}%,loss:{round(total_loss / i, 4)}")
        temp_acc = round((total_acc / count) * 100, 2)
        TACC.append(temp_acc)
        if temp_acc > best_acc:
            best_acc = temp_acc
 
    print(f"LOSS_array:{LOSS}")
    print(f"ACC_array:{ACC}")
    print(f"TACC_array:{TACC}")
