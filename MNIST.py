from torch.utils.data import Dataset
import torch
import os
import struct
import numpy as np

class MNIST(Dataset):#need implement __len__(),__getitem__():
    def __init__(self,root,train=True,transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        if self.train == True:
            self.kind = "train"
        else:
            self.kind = "t10k"

        self.labels_path = os.path.join(root,
                               '%s-labels.idx1-ubyte'
                               % self.kind)
        self.images_path = os.path.join(root,
                               '%s-images.idx3-ubyte'
                               % self.kind)
        with open(self.labels_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II',
                                    lbpath.read(8))
            self.labels = np.fromfile(lbpath,
                                     dtype=np.uint8)
        self.labels = torch.LongTensor(self.labels)
        with open(self.images_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
            self.images = np.fromfile(imgpath,
                                      dtype=np.uint8).reshape((len(self.labels), 28,28))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,index):
        img,target = self.images[index][:,:,np.newaxis],self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        sample = {'img': img, 'target': target}
        return sample

def main():
    import matplotlib.pyplot as plt
    train = MNIST(root='./MNIST_data/',train=True,transform=None)
    print(len(train))
    for (cnt,i) in enumerate(train):
        image = i['img']
        label = i['target']
        print(image.shape)
        ax = plt.subplot(4, 4, cnt+1)
        # ax.axis('off')
        ax.imshow(image)
        ax.set_title(label)
        plt.pause(0.001)
        if cnt ==15:
            break
    a = input()
if __name__ == '__main__':
	main()


