import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import torch.optim as optim
import torch.nn.functional as F

from MNIST import MNIST
from CNN import *

# Training settings
batch_size = 60000
lr = 0.001


# MNIST Dataset

train_dataset = MNIST(root='./MNIST_data/',train=True,transform=transforms.ToTensor())
test_dataset = MNIST(root='./MNIST_data/',train=False,transform=transforms.ToTensor())

# Data Loader

train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)

#model 

net = CNN()
optimizer = optim.Adam(net.parameters(),lr=lr)
lossfunc = nn.CrossEntropyLoss()


def main():
	# 如果模型文件存在则尝试加载模型参数
    if os.path.exists('./model.pth'):
        try:
            net.load_state_dict(torch.load('./model.pth'))
        except Exception as e:
            print(e)
            print("Parameters Error")
    for epoch in range(1, 100):
        train(epoch,net,optimizer,lossfunc,train_loader)
        torch.save(net.state_dict(),'./model.pth')
        test(net,lossfunc,test_loader)
    '''
    dataloader = DataLoader(train_dataset, batch_size=50,shuffle=True)
    for step, i in enumerate(dataloader):
        b_x = i['img'].shape
        b_y = i['target'].shape
        print ('Step: ', step, '| train_data的维度' ,b_x,'| train_target的维度',b_y)
    '''
if __name__ == '__main__':
	main()
