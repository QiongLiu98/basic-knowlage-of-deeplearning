# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 21:54:59 2019

@author: ASUS
"""

#  首先当然肯定要导入torch和torchvision，至于第三个是用于进行数据预处理的模块
import torch
import torchvision
import torchvision.transforms as transforms
 
#  **由于torchvision的datasets的输出是[0,1]的PILImage，所以我们先先归一化为[-1,1]的Tensor**
    #  首先定义了一个变换transform，利用的是上面提到的transforms模块中的Compose( )
    #  把多个变换组合在一起，可以看到这里面组合了ToTensor和Normalize这两个变换
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) 
 
    # 定义了我们的训练集，名字就叫trainset，至于后面这一堆，其实就是一个类：
    # torchvision.datasets.CIFAR10( )也是封装好了的，就在我前面提到的torchvision.datasets
    # 模块中,不必深究，如果想深究就看我这段代码后面贴的图1，其实就是在下载数据
    #（不翻墙可能会慢一点吧）然后进行变换，可以看到transform就是我们上面定义的transform
trainset = torchvision.datasets.CIFAR10(root='E:/waterloo/opencv_learning', train=True,
                                        download=True, transform=transform)
    # trainloader其实是一个比较重要的东西，我们后面就是通过trainloader把数据传入网
    # 络，当然这里的trainloader其实是个变量名，可以随便取，重点是他是由后面的
    # torch.utils.data.DataLoader()定义的，这个东西来源于torch.utils.data模块，
    #  网页链接http://pytorch.org/docs/0.3.0/data.html，这个类可见我后面图2
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
    # 对于测试集的操作和训练集一样，我就不赘述了
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)
    # 类别信息也是需要我们给定的
classes = ('plane', 'car', 'bird', 'cat',
'deer', 'dog', 'frog', 'horse', 'ship', 'truck')