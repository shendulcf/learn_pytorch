from PIL import Image
import torch
import torchvision 
import torch.nn as nn
# from src.P20_use_GPU import FanNet,fanfan
import sys
# sys.path.append(r"D:\Workspace\Study_Pytorch\xiaotudui")

img_path = r"data_test\plane.jpg"
image =  Image.open(img_path)

class FanfNet(nn.Module):
    
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
class FanNet(nn.Module):
    
  def __init__(self):
      super(FanNet,self).__init__()
      self.model = nn.Sequential(
          nn.Conv2d(3, 32, 5, padding=2),
          nn.MaxPool2d(2),
          nn.Conv2d(32, 32, 5, padding=2),
          nn.MaxPool2d(2),
          nn.Conv2d(32, 64, 5, padding=2),
          nn.MaxPool2d(2),
          nn.Flatten(),
          nn.Linear(1024,64),
          nn.Linear(64,10),
      )

  def forward(self,x):
    x = self.model(x)
    return x


transfrom = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((32,32)),
        torchvision.transforms.ToTensor()
    ]
)

model = torch.load( 'src/fanfan_9.pt', map_location=torch.device('cpu'))
print(model)

img = transfrom(image)
img = torch.reshape(img,(1,3,32,32))
# img = img.reshape(1,3,32,32)
model.eval()
with torch.no_grad():
    output = model(img)
print(output)
print(output.argmax())