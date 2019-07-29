# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 03:42:04 2019

@author: liuqiong
"""
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose(
    [transforms.ToTensor()])

# 训练集
trainset = torchvision.datasets.MNIST(root='./data',     # 选择数据的根目录
                                      train=True,
                                      download=False,    # 不从网络上download图片
                                      transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                         shuffle=True, num_workers=2)
# 测试集
testset = torchvision.datasets.MNIST(root='./data',     # 选择数据的根目录
                                     train=False,
                                     download=False,    # 不从网络上download图片
                                     transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                        shuffle=False, num_workers=2)

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

# 选择一个 batch 的图片
dataiter = iter(trainloader)
images, labels = dataiter.next()

# 显示图片
imshow(torchvision.utils.make_grid(images))
plt.show()
# 打印 labels
print(' '.join('%11s' % labels[j].numpy() for j in range(4)))
print(trainset.train_data.size())
print(testset.test_data.size())

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.rnn = nn.LSTM(          # 使用 LSTM 结构
            input_size = 28,         # 输入每个元素的维度，即图片每行包含 28 个像素点
            hidden_size = 84,        # 隐藏层神经元设置为 84 个
            num_layers=1,            # 隐藏层数目，单层
            batch_first=True,        # 是否将 batch 放在维度的第一位，(batch, time_step, input_size)
        )
        self.out = nn.Linear(84, 10) # 输出层，包含 10 个神经元，对应 0～9 数字

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)   
        # 选择图片的最后一行作为 RNN 输出
        out = self.out(r_out[:, -1, :])
        return out