from torch.utils.data import Dataset
from PIL import Image
import os

# img_path = r'D:\workspace\Study_Pytorch\小土堆\dataset\dataset1\train\ants\6743948_2b8c096dda.jpg'
# img = Image.open(img_path)
# img.show()



class Mydata(Dataset):

    def __init__(self, root_dir, label_dir):

        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(root_dir, label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx) :
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label
    
    def __len__(self):
        return len(self.img_path)

root_dir = r'E:\workspace\Study_Pytorch\xiaotudui\dataset\dataset1\train'
ants_label_dir = 'ants'
bees_label_dir = 'bees'
ants_dataset = Mydata(root_dir, ants_label_dir)
bees_dataset = Mydata(root_dir, bees_label_dir)

train_dataset = ants_dataset + bees_dataset # 可以用来制作子数据集或者进行数据集的扩充

img, label = bees_dataset[1]
img.show()
print(label)

