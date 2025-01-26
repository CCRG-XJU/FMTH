from config import *
from torchvision.transforms import Resize, Grayscale, Normalize

class Loader(Dataset):
    def __init__(self, img_dir, txt_dir, NB_CLS=None, dname='imagenet'):
        self.img_dir = img_dir
        self.file_list = np.loadtxt(txt_dir, dtype='str')
        self.NB_CLS = NB_CLS
        self.dname = dname
        if self.dname == 'imagenet':
            self.imagenet_resize = Resize((256, 256))
            # self.imagenet_Grayscale = Grayscale(num_output_channels=3)
            # self.Norm = Normalize(mean=T.as_tensor([0.485, 0.456, 0.406]), std=T.as_tensor([0.229, 0.224, 0.225]))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if T.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir,
                                self.file_list[idx][0])
        if self.dname == 'imagenet':
            image = Image.open(img_name).convert('RGB')
        else:
            image = Image.open(img_name)

        if self.NB_CLS != None:
            if len(self.file_list[idx])>2:
                label = [int(self.file_list[idx][i]) for i in range(1,self.NB_CLS+1)]
                # label = T.tensor(label, dtype=T.float)
                # print(label)
                label = T.FloatTensor(label)
                # label = T.from_numpy(label).float()
            else:
                label = int(self.file_list[idx][1])
            if self.dname == 'imagenet':
                image = self.imagenet_resize(image)
                # image = self.imagenet_Grayscale(image)
                # image = transforms.ToTensor()(image)
                # image = self.Norm(image)
            image = transforms.ToTensor()(image)

            return image, label
        else:
            if self.dname == 'imagenet':
                image = self.imagenet_resize(image)
                # image = self.imagenet_Grayscale(image)
            image = transforms.ToTensor()(image)

            return image
        
class Loader_imagenet(Dataset):
    def __init__(self, img_dir, txt_dir, NB_CLS=None, dname='imagenet'):
        self.img_dir = img_dir
        self.file_list = np.loadtxt(txt_dir, dtype='str')
        self.NB_CLS = NB_CLS
        self.dname = dname
        if self.dname == 'imagenet':
            self.imagenet_resize = Resize((256, 256))
            # self.imagenet_Grayscale = Grayscale(num_output_channels=3)
            # self.Norm = Normalize(mean=T.as_tensor([0.485, 0.456, 0.406]), std=T.as_tensor([0.229, 0.224, 0.225]))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if T.is_tensor(idx):
            idx = idx.tolist()

        # img_dir = self.img_dir + '/' + self.file_list[idx][0].split("_")[0]
        # print(img_dir)
        img_name = os.path.join(self.img_dir,
                                self.file_list[idx][0])
        # image = Image.open(img_name)
        image = Image.open(img_name).convert('RGB')

        if self.NB_CLS != None:
            if len(self.file_list[idx])>2:
                label = [int(self.file_list[idx][i]) for i in range(1,self.NB_CLS+1)]
                # label = T.tensor(label, dtype=T.float)
                # print(label)
                label = T.FloatTensor(label)
                # label = T.from_numpy(label).float()
            else:
                label = int(self.file_list[idx][1])
            if self.dname == 'imagenet':
                image = self.imagenet_resize(image)
                # image = self.imagenet_Grayscale(image)
                # image = transforms.ToTensor()(image)
                # image = self.Norm(image)
            image = transforms.ToTensor()(image)

            return image, label
        else:
            if self.dname == 'imagenet':
                image = self.imagenet_resize(image)
                # image = self.imagenet_Grayscale(image)
                # image = self.Norm(image)
            image = transforms.ToTensor()(image)

            return image
        
def image_transform3(resize_size=256, crop_size=224, dname='imagenet', data_set="train_set"):
    return transforms.Compose([
            transforms.Resize(resize_size, interpolation=Image.BICUBIC),
            transforms.RandomRotation(15),  # 添加随机旋转
            transforms.RandomHorizontalFlip(),  # 添加随机水平翻转
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]) if data_set == "train_set" else transforms.Compose([
            transforms.Resize((crop_size, crop_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

class Loader3(Dataset):
    def __init__(self, img_dir, txt_dir, NB_CLS=None, dname='imagenet', resize_size=256, crop_size=224, data_set="train_set"):
        self.img_dir = img_dir
        self.file_list = np.loadtxt(txt_dir, dtype='str')
        self.NB_CLS = NB_CLS
        self.dname = dname
        # self.image_transform = image_transform2(resize_size=resize_size, crop_size=crop_size, data_set=data_set)
        self.image_transform = image_transform3(resize_size=resize_size, crop_size=crop_size, dname=dname, data_set=data_set)
        # if self.dname == 'imagenet':
        #     self.imagenet_resize = Resize((256, 256))
        #     self.imagenet_Grayscale = Grayscale(num_output_channels=3)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if T.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir,
                                self.file_list[idx][0])
        image = Image.open(img_name).convert('RGB')

        if self.NB_CLS != None:
            if len(self.file_list[idx])>2:
                label = [int(self.file_list[idx][i]) for i in range(1,self.NB_CLS+1)]
                # label = T.tensor(label, dtype=T.float)
                # print(label)
                label = T.FloatTensor(label)
                # label = T.from_numpy(label).float()
            else:
                label = int(self.file_list[idx][1])
            # if self.dname == 'imagenet':
            #     image = self.imagenet_resize(image)
            #     image = self.imagenet_Grayscale(image)
            # image = transforms.ToTensor()(image)
            image = self.image_transform(image)

            return image, label
        else:
            # if self.dname == 'imagenet':
            #     image = self.imagenet_resize(image)
            #     image = self.imagenet_Grayscale(image)
            # image = transforms.ToTensor()(image)
            image = self.image_transform(image)
            return image