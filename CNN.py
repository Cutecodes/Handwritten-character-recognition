import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=10,kernel_size=5), #stride=1,padding=2)
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(10)# out_channels
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=10,out_channels=20,kernel_size=5),#stride=1,padding=2),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(20)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=20,out_channels=40,kernel_size=3),#stride=1,padding=2),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(40)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(40,80),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(80,20),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(20,10)

    def forward(self,x):
        in_size = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(in_size,-1)
        #print(x.size)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
   
def train(epoch,model,optimizer,lossfunc,data_loader):
    for batch_idx,data in enumerate(data_loader):
        inputdata = data['img']
        target = data['target']
        output = model(inputdata)
        loss = lossfunc(output,target)

        if batch_idx % 200 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(inputdata), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.data))
    optimizer.zero_grad() #梯度清零
    loss.backward()       #反向传播
    optimizer.step()      #使用optimizer进行梯度下降

def test(model,lossfunc,data_loader):
    test_loss = 0
    correct = 0
    for data in data_loader:
        inputdata,target = Variable(data['img'],volatile=True),Variable(data['target'])
        output = model(inputdata)
        test_loss += lossfunc(output,target).data
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    #test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))

