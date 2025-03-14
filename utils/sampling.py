'''
sampling采样-->拆分数据集为 non-iid 数据
'''

import numpy as np
np.random.seed(1)
from torchvision import datasets, transforms
# import matplotlib.pyplot as plt
import torch
import random
random.seed(1)

def dataset_iid(dataset, num_users) -> dict:
    datasize = len(dataset) // num_users  # 指定每个user或者说设备多少块数据
    all_idxs = [i for i in range(len(dataset))]  # all_idxs是dataset分块数据后每块数据的index
    # 下面就是先 shuffle all_idxs，把dataset分块数据的index打乱
    random.shuffle(all_idxs)
    # 然后把打乱的index按顺序和num_items（每块大小）分配给各个client
    dict_users = {}
    for i in range(num_users):
        idx_begin = i * datasize
        dict_users[i] = np.array(all_idxs[(idx_begin):(idx_begin+datasize)])
    return dict_users


def dataset_noniid(dataset, num_users, num_classes, main_label_prop, sageflow=False):
    dict_users = {i: np.array([], dtype='int32') for i in range(num_users)}
    all_idxs = np.arange(len(dataset))  # all_idxs:  [0,------------,59999]    idx代表图片在原始数据集中的索引
    labels = dataset.targets if num_classes < 200 else [i[1] for i in dataset.images]
    labels = labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels) # 获取标签

    # sort labels
    idxs_labels = np.vstack((all_idxs, labels))  # 把索引（60000个）和标签进行堆叠匹配
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()] # argsort()返回的是数组索引
    idxs = idxs_labels[0, :]
    now_idx, dict_class = {i:0 for i in range(num_classes)}, {i: [] for i in range(num_classes)}
    for idx, label in zip(idxs_labels[0,:], idxs_labels[1,:]):
        dict_class[label].append(idx)

    # label_size
    main = 1 if num_classes<num_users else num_classes // num_users
    other = num_classes - main
    min_length = len(min(dict_class.values(), key=lambda p: len(p))) * num_classes // num_users # 每个user的数据集量
    main_label_size = int(np.floor(min_length * main_label_prop)) // main #一个主类的样本量
    other_label_size = (min_length - main_label_size * main) // other #一个其他类的样本量

    for i in range(num_users):
        c = i % num_classes
        main_label = np.array([main*c + j for j in range(main)])  # 选一个为主类
        other_label = np.delete(np.arange(num_classes),main_label)
        # main label idx array
        for j in main_label:
            idx_begin = now_idx[j]
            dict_users[i] = np.append(dict_users[i], np.array(dict_class[j][idx_begin : idx_begin+main_label_size],dtype='int32'), axis=0)
            now_idx[j] += main_label_size
        # other label idx array
        for j in other_label:
            idx_begin = now_idx[j]
            dict_users[i] = np.append(dict_users[i], np.array(dict_class[j][idx_begin : idx_begin+other_label_size],dtype='int32'), axis=0)
            now_idx[j] += other_label_size

    # if sageflow:
    #     dict_common = dict_users[0]
    #     for i in range(num_users):
    #         dict_users[i] = dict_users[i+1]
    #     del dict_users[num_users]
    #     return dict_users, dict_common

    return dict_users


def mnist_noniid_modified_Vtest(dataset, num_users, min_train=5900, max_train=6100, main_label_prop=0.8, other=5):
    """
    non-i.i.d数据生成
    100个client
    数量分布随机从 min_train 到 max_train
    每个 client 80% 数据为一类图片， 20% 为其他类图片
    返回的dict_users最终要放到 torch 的 dataloader里面进行数据加载 (non-iid)
    """
    # random_seed = np.random.randint(low=30000, high=40000)
    # print("random_seed: ", random_seed)
    # np.random.seed(random_seed)  # 之前固定了0
    np.random.seed(0)

    num_shards, num_imgs = 10, 6000  # 10类图片，每类6000张

    # 根据num_users（users数量）创建dict_users
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)     # 返回一个有终点和起点的固定步长的排列 [0, 1, 2,......, (num_shards*num_imgs-1)]
    labels = dataset.targets.numpy()       # 变为numpy.ndarray类型的labels（其实就是一堆int数据）
    print("labels:\n", labels)
    print("len: ", len(labels))
    print(type(labels))

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # 每个user，也就是每个devices，
    for i in range(num_users):
        datasize = np.random.randint(min_train, max_train + 1)  # 随机数量
        main_label = np.random.randint(0, 10)  # 0-9随机选一个为主类
        print("user: %d, data_size: %d, main_label: %d" % (i, datasize, main_label))

        main_label_size = int(np.floor(datasize * main_label_prop))
        other_label_size = datasize - main_label_size
        # print("user: %d, main_label_size: %d, other_label_size: %d" % (i, main_label_size, other_label_size))

        # main label idx array
        idx_begin = np.random.randint(0, num_imgs - main_label_size) + main_label * num_imgs
        # print("idx_begin: %d, begin class: %d, end class: %d" %(idx_begin, idxs_labels[1][idx_begin], idxs_labels[1][idx_begin + main_label_size]))
        dict_users[i] = np.concatenate((dict_users[i],      # concatenate, axis=0 --> 上下拼接，dict_users在上，idxs在下
                                        idxs[idx_begin:(idx_begin+main_label_size)]), axis=0)

        # other label idx array
        other_label_dict = np.zeros(other_label_size, dtype='int64')

        other_nine_label = np.delete(np.arange(10), main_label)


        other_label_class = np.random.choice(other_nine_label, size=other, replace=False)

        count = 0

        for j in range(other_label_size):
            label = other_label_class[count % other]
            other_label_dict[j] = idxs[int(np.random.randint(0, num_imgs) + label * num_imgs)]
            count += 1

        dict_users[i] = np.concatenate((dict_users[i], other_label_dict), axis=0)

        # for k in range(datasize):
        #     idx = dict_users[i][k]
        #     print("idx: %d, label: %d" %(dict_users[i][k], labels[idx]))
        #
        # print("++++++++++++++++++++++++++++++++++++++")

    return dict_users





def cifar_noniid(dataset, num_users, min_train=5900, max_train=6100, main_label_prop=0.8, other=9, map_file=None):
    """
    non-i.i.d数据生成

    100个client

    数量分布随机从min_train到max_train

    每个client 80%数据为一类图片， 20%为其他类图片

    """
    # random_seed = np.random.randint(low=30000, high=40000)
    # print("random_seed: ", random_seed)
    # np.random.seed(random_seed)  # 之前固定了0
    np.random.seed(0)

    num_shards, num_imgs = 10, 6000  # 10类图片，每类6000张

    # min_train = 200  # 最少200张
    # max_train = 1000  # 最多1000张
    # main_label_prop = 0.8  # 80%来自同一张图片，20%均匀来自其他9类图片

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)

    # labels = np.array(dataset.targets)  # 修改dataset.target_transform无法解决
    labels = np.array(dataset.targets)   # 上面的labels改为这一行的labels（模仿mnist）
    print("labels:\n", labels)
    print("len: ", len(labels))
    print(type(labels))

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # idxs:           [--------------------]    idx代表图片在原始数据集中的索引
    # idxs_labels[1]: [0, 0, 0, ... 9, 9, 9]    label代表图片对应的数字标签

    # df_path = conf.DATASET_PATH + 'decision_making_dataset/MAP_ID2DataSize_Fairness2022V1/map_id_to_datasize_iid-original.csv'
    # df = pd.DataFrame(pd.read_csv(df_path), index=None)

    for i in range(num_users):
        # 这里的随机数量，要修改成指定的数量-->有num_users个设备(users)，对应index的那个user指定一个datasize（通过读取文件实现）
        datasize = np.random.randint(min_train, max_train + 1)  # 随机数量
        # datasize = int(map_file.iloc[i, map_file.columns.get_loc('datasize')])
        # main_label = int(map_file.iloc[i, map_file.columns.get_loc('main_label')])    # 已经确定好的
        main_label = i  # 0-9随机选一个为主类
        print("user: %d, data_size: %d, main_label: %d" % (i, datasize, main_label))
        # df.loc[i, ['main_label']] = main_label

        main_label_size = int(np.floor(datasize * main_label_prop))
        other_label_size = datasize - main_label_size
        # print("user: %d, main_label_size: %d, other_label_size: %d" % (i, main_label_size, other_label_size))

        # main label idx array
        idx_begin = np.random.randint(0, num_imgs - main_label_size) + main_label * num_imgs
        # print("idx_begin: %d, begin class: %d, end class: %d" %(idx_begin, idxs_labels[1][idx_begin], idxs_labels[1][idx_begin + main_label_size]))
        dict_users[i] = np.concatenate((dict_users[i],
                                        idxs[(idx_begin):(idx_begin+main_label_size)]), axis=0)

        # other label idx array
        other_label_dict = np.zeros(other_label_size, dtype='int64')

        other_nine_label = np.delete(np.arange(10), main_label)


        other_label_class = np.random.choice(other_nine_label, size=other, replace=False)

        count = 0

        for j in range(other_label_size):
            label = other_label_class[count % other]
            other_label_dict[j] = idxs[int(np.random.randint(0, num_imgs) + label * num_imgs)]
            count += 1

        dict_users[i] = np.concatenate((dict_users[i], other_label_dict), axis=0)
    # df.to_csv(df_path, index=None)
    return dict_users


# def imshow(img):
#     img = img / 2 + 0.5  # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))  #transpose()作用是调换数组行列值的索引值


if __name__ == '__main__':

    num_users = 100

    dataset_train = datasets.MNIST('../data/mnist', train=True, download=False,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    dict_users = dataset_noniid(dataset_train, num_users,10,0.8,9)
    # dict_users = mnist_noniid_modified(dataset_train, num_users)

    # np.save('../simulation/dataset_noniid_200_1000.npy', dict)
    # cs = np.load('../simulative_client_state.npy')
    # print(cs[0])


    trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)

    trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=16, shuffle=True)

    # 类别标签
    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # dict_users = cifar_iid(dataset_train, num_users)
    # dict_users = cifar_noniid(dataset_train, num_users)
    for k, v in dict_users.items():
        if k == 92: 
            print(k)
        print(k,len(v))

    # # 获取随机数据
    # dataiter = iter(trainloader)  # 把trainloader变成一个可迭代的对象
    # images, labels = dataiter.next()
    #
    # # 展示图像
    # imshow(torchvision.utils.make_grid(images))
    # # 显示图像标签
    # print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    #
    # plt.show()




