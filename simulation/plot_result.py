import numpy as np
import matplotlib.pyplot as plt

acc_random_list = []
acc_fedcs_list  = []
acc_ucb_list    = []
acc_linucb_list = []

min_list        = []
max_list        = []

length = 200
interval = 7

for i in range(length):
    if i % interval == 0:
        acc_random_i = acc_random[i]
        acc_fedcs_i  = acc_fedcs[i]
        acc_ucb_i    = acc_ucb[i]
        acc_linucb_i = acc_linucb[i]

        # 存储数据
        acc_random_list.append(acc_random_i)
        acc_fedcs_list.append(acc_fedcs_i)
        acc_ucb_list.append(acc_ucb_i)
        acc_linucb_list.append(acc_linucb_i)

        # 找到最大最小值
        compare_list = [acc_random_i, acc_fedcs_i, acc_ucb_i]
        min_list.append(min(compare_list))
        max_list.append(max(compare_list))


round = range(length)
rount_interval = np.arange(0, length, interval)

# plt.title('mnist noniid prob: 0.8')
# plt.plot(round, acc_random[:length], label='random')
# plt.plot(round, acc_fedcs[:length], label='fedcs')
# plt.plot(round, acc_ucb[:length], label='ucb')
# plt.plot(round, acc_linucb[:length], label='linucb')

# plt.plot(rount_interval, acc_random_list, 'v-', label='random')
# plt.plot(rount_interval, acc_fedcs_list, '*-', label='fedcs')
# plt.plot(rount_interval, acc_ucb_list, '+--', label='ucb')
plt.xticks([0, 50, 100, 150, 200])
plt.plot(rount_interval, acc_linucb_list, 'v--', linewidth=1.5, label='linucb')
plt.fill_between(rount_interval, min_list, max_list, color='#ff7f0e', alpha=0.25)
plt.tick_params(labelsize=18)
plt.xlabel('FL Rounds')
plt.ylabel('Test Accuracy')
# plt.ylim(0, 60)
plt.legend()

from fedLearnSim import configuration as conf

plt.savefig(fig_savepath)   # 新增 Jan 18

plt.show()
