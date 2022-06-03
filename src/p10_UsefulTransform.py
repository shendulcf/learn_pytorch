from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")

img_path = r'E:\Workspace\Study_Pytorch\xiaotudui\dataset\dataset1\train\ants\82852639_52b7f7f5e3.jpg'
img = Image.open(img_path)

# ----> ToTensor
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
writer.add_image("tensor_img", tensor_img)

# ----> Nomalize
trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
norm_img = trans_norm(tensor_img)
writer.add_image("normlize_img", norm_img,2)

# ----> Resize
print(img.size)
trans_resize = transforms.Resize((512,512))
resize_img = trans_resize(tensor_img)
writer.add_image("resize_img", resize_img,3)

# ----> RandomRotation
trans_randomrotation = transforms.RandomRotation(degrees=30)
randomrotation_img = trans_randomrotation(tensor_img)
writer.add_image("randomrotation_img", randomrotation_img)


# ----> Compose 传入的是一个transform参数列表
trans_compose = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])
compose_img = trans_compose(img)
writer.add_image("compose_img", compose_img)

# ----> RandomCrop
randomcrop = transforms.RandomCrop(124)
crop_img = randomcrop(tensor_img)
writer.add_image("crop_img", crop_img)

writer.close()