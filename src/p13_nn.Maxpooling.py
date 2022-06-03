from re import X
import  torch
from torch import nn
from torch.nn import MaxPool2d

# 最大池化的目的就是在减少数据量的同时尽量的保留特征

input = torch.tensor([[1,2,0,3,1],[0,1,2,3,1],[1,2,1,0,0],[5,2,3,1,1],[2,1,0,1,1]],dtype=torch.float32)
input = input.reshape(-1,1,5,5)

class Fanfan(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.maxpool1 = MaxPool2d(3,ceil_mode=False)

    def forward(self,x):
        x = self.maxpool1(x)
        return x

fanfan = Fanfan()
output = fanfan(input)
print(output)