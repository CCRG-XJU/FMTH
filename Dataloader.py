from config import *
from torchvision.transforms import Resize, Grayscale

class Loader(Dataset):
    def __init__(self, img_dir, txt_dir, NB_CLS=None, dname='imagenet'):
        self.img_dir = img_dir
        self.file_list = np.loadtxt(txt_dir, dtype='str')
        self.NB_CLS = NB_CLS
        self.dname = dname
        if self.dname == 'imagenet':
            self.imagenet_resize = Resize((256, 256))
            self.imagenet_Grayscale = Grayscale(num_output_channels=3)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if T.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir,
                                self.file_list[idx][0])
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
                image = self.imagenet_Grayscale(image)
            image = transforms.ToTensor()(image)

            return image, label
        else:
            if self.dname == 'imagenet':
                image = self.imagenet_resize(image)
                image = self.imagenet_Grayscale(image)
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
            self.imagenet_Grayscale = Grayscale(num_output_channels=3)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if T.is_tensor(idx):
            idx = idx.tolist()

        # img_dir = self.img_dir + '/' + self.file_list[idx][0].split("_")[0]
        # print(img_dir)
        img_name = os.path.join(self.img_dir,
                                self.file_list[idx][0])
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
                image = self.imagenet_Grayscale(image)
            image = transforms.ToTensor()(image)

            return image, label
        else:
            if self.dname == 'imagenet':
                image = self.imagenet_resize(image)
                image = self.imagenet_Grayscale(image)
            image = transforms.ToTensor()(image)

            return image