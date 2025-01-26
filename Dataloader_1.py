from config import *
import torch

class Loader(Dataset):
    def __init__(self, img_dir, txt_dir, NB_CLS=None):
        self.img_dir = img_dir
        self.file_list = np.loadtxt(txt_dir, dtype='str')
        self.NB_CLS = NB_CLS

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
            return transforms.ToTensor()(image), label
        else:
            return transforms.ToTensor()(image)

class MyCIFAR10(dsets.CIFAR10):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img = transforms.Resize(224)(img)
        img = transforms.ToTensor()(img)
        target = np.eye(10, dtype=np.int8)[np.array(target)]
        target = T.FloatTensor(target)
        return img, target


def cifar_dataset(batch_size, num_workers):

    train_size = 500
    test_size = 100

    cifar_dataset_root = './data'
    # Dataset
    train_dataset = MyCIFAR10(root=cifar_dataset_root,
                              train=True,
                              transform=None,
                              download=True)

    test_dataset = MyCIFAR10(root=cifar_dataset_root,
                             train=False,
                             transform=None)

    database_dataset = MyCIFAR10(root=cifar_dataset_root,
                                 train=False,
                                 transform=None)

    X = np.concatenate((train_dataset.data, test_dataset.data))
    L = np.concatenate((np.array(train_dataset.targets), np.array(test_dataset.targets)))

    first = True
    for label in range(10):
        index = np.where(L == label)[0]

        N = index.shape[0]
        perm = np.random.permutation(N)
        index = index[perm]

        if first:
            test_index = index[:test_size]
            train_index = index[test_size: train_size + test_size]
            database_index = index[train_size + test_size:]
        else:
            test_index = np.concatenate((test_index, index[:test_size]))
            train_index = np.concatenate((train_index, index[test_size: train_size + test_size]))
            database_index = np.concatenate((database_index, index[train_size + test_size:]))
        first = False

    train_dataset.data = X[train_index]
    train_dataset.targets = L[train_index]
    test_dataset.data = X[test_index]
    test_dataset.targets = L[test_index]
    database_dataset.data = X[database_index]
    database_dataset.targets = L[database_index]

    print("train_dataset", train_dataset.data.shape[0])
    print("test_dataset", test_dataset.data.shape[0])
    print("database_dataset", database_dataset.data.shape[0])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_workers)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=num_workers)

    database_loader = torch.utils.data.DataLoader(dataset=database_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=num_workers)

    return train_loader, test_loader, database_loader
