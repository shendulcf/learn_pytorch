# ----> GPU的使用
# 网络模型
# 数据的（输入和标注）
# 损失函数
# .cuda() or .to(device) 后者比较方便


import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
import time
import os

# ----> gpu设置
# device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

## 查看GPU讯息
print(torch.cuda.is_available() )# cuda是否可用
print(torch.cuda.device_count() )# gpu数量
print(torch.cuda.current_device())# 当前设备索引, 从0开始
print(torch.cuda.get_device_name(0))# 返回gpu名字
## 使用GPU
# device = torch.cuda.current_device() if args.cuda else torch.device('cpu')
os.environ['CUDA_VISIBLE_DEVICES']='0' # 设置程序可见的gpu的环境变量
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# ----> dataset

train_data = torchvision.datasets.CIFAR10(root='../data',train=True,transform=torchvision.transforms.ToTensor(),
                                            download=True)
test_data = torchvision.datasets.CIFAR10(root='../data',train=False,transform=torchvision.transforms.ToTensor(),
                                            download=True)

# ----> len dataset
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练集的长度：{}".format(train_data_size))
print("测试集的长度：{}".format(test_data_size))

# ----> Dataloader
from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=64)

# ----> model set

class FanNet(nn.Module):
    
    def __init__(self):
        super(FanNet,self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 32, 5, padding=2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, 5, padding=2)
        self.maxpool3 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(1024, 64)
        self.linear2 = nn.Linear(64, 10)

    def forward(self,x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x

fanfan = FanNet()
# fanfan.cuda()
fanfan = fanfan.to(device)


# ----> Train
start_time = time.time() # .time()记录当前时间

# Loss
loss1 = nn.CrossEntropyLoss()
# optimer
learning_rate = 0.01
optimer = torch.optim.Adam(fanfan.parameters(),lr=learning_rate)

# train_model parameters setting
epoch = 10
total_train_step = 0
total_test_step = 0

# tensorboard start
writer = SummaryWriter('../loss_train')

for i in range(epoch):
    print("----------第{}轮训练开始----------".format(i))
    
    ## train start
    # fanfan.train() 加与不加都可以，只对一些特定的层有作用
    for img, label in train_dataloader:
        img = img.to(device)
        label = label.to(device)
        output = fanfan(img)
        loss = loss1(output,label).requires_grad_(True)

        ### 开始优化
        optimer.zero_grad()
        loss.backward()
        optimer.step()

        total_train_step += 1
        if total_train_step % 50 == 0:
            end_time = time.time()
            print("训练100次需要的时间:{}".format(end_time - start_time))
            print("训练次数:{}, loss:{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)


    # test_start 更准确的说是验证，计算的是整个数据集的loss
    fanfan.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for img, label in test_dataloader:
            img = img.to(device)
            label = label.to(device)
            output = fanfan(img)
            loss = loss1(output, label)
            optimer.zero_grad()
            loss.backward()
            optimer.step()
            total_test_step += 1
            total_test_loss += loss
            accuracy = (output.argmax() == label).sum()
            total_accuracy += accuracy

    print("整体测试集上的Loss:{}".format(total_test_loss))
    print("整体测试集上的Accura:{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracu", total_accuracy/test_data_size, total_test_step)
    total_test_step += 1

    torch.save(fanfan, "fanfan_{}.pt".format(i))
    print("model save already")
