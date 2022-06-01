from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")

img_path = r'xiaotudui\dataset\dataset1\train\bees\21399619_3e61e5bb6f.jpg'
img = Image.open(img_path)

# ----> ToTensor
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
writer.add_image("tensor_img",tensor_img)

# ----> Nomalize
trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
norm_img = trans_norm(tensor_img)
writer.add_image("normlize_img",norm_img,2)

# ----> Resize
print(img.size)
trans_resize = transforms.Resize((512,512))
resize_img = trans_resize(tensor_img)
writer.add_image("resize_img",resize_img,3)




Transform = transforms.Compose([



])