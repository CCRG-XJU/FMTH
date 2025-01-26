from models_fusion import *
from torch.optim.lr_scheduler import CosineAnnealingLR
from loss import *
from Dataloader_1 import *
from tqdm import tqdm
import visdom
import time
import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from Retrieval1 import DoRetrieval5, DoRetrieval_cifar3, save_mat



class Visualizer(object):
    """
    封装了visdom的基本操作，但是你仍然可以通过`self.vis.function`
    调用原生的visdom接口
    """

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, use_incoming_socket=False, **kwargs)

        # 画的第几个数，相当于横座标
        # 保存（’loss',23） 即loss的第23个点
        self.index = {}
        self.log_text = ''
        self.env = env
    
    def save(self):
        self.vis.save([self.env])
 
    def reinit(self, env='default', **kwargs):
        """
        修改visdom的配置
        """
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        """
        一次plot多个
        @params d: dict (name,value) i.e. ('loss',0.11)
        """
        for k, v in d.items():
            self.plot(k, v)

    def img_many(self, d):
        for k, v in d.items():
            self.img(k, v)

    def plot(self, name, y, **kwargs):
        """
        self.plot('loss',1.00)
        """
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name, opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1

    def img(self, name, img_, **kwargs):
        """
        self.img('input_img',t.Tensor(64,64))
        self.img('input_imgs',t.Tensor(3,64,64))
        self.img('input_imgs',t.Tensor(100,1,64,64))
        self.img('input_imgs',t.Tensor(100,3,64,64),nrows=10)
        ！！！don‘t ~~self.img('input_imgs',t.Tensor(100,64,64),nrows=10)~~！！！
        """
        self.vis.images(img_.cpu().numpy(),
                        win=name,
                        opts=dict(title=name),
                        **kwargs
                        )

    def log(self, info, win='log_text'):
        """
        self.log({'loss':1,'lr':0.0001})
        """

        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'),
            info=info))
        self.vis.text(self.log_text, win)

    def __getattr__(self, name):
        return getattr(self.vis, name)

def get_args_parser():
    parser = argparse.ArgumentParser('DHD', add_help=False)

    parser.add_argument('--gpu_id', default="1", type=str, help="""Define GPU id.""")
    parser.add_argument('--data_dir', default="/data", type=str, help="""Path to dataset.""")
    parser.add_argument('--dataset', default="cifar-10", type=str, help="""Dataset name: imagenet, nuswide_m, coco.""")
    parser.add_argument('--accumulation_steps', default=4, type=int, help="""accumulation_steps""")
    parser.add_argument('--batch_size', default=64, type=int, help="""Training mini-batch size.""")
    parser.add_argument('--num_workers', default=16, type=int, help="""Number of data loading workers per GPU.""")
    parser.add_argument('--encoder', default="SwinT", type=str, help="""Encoder network: ResNet, AlexNet, ViT, DeiT, SwinT.""")
    parser.add_argument('--N_bits', default='16, 32, 64', type=str, help="""Number of bits to retrieval.""")
    parser.add_argument('--init_lr', default=3e-4, type=float, help="""Initial learning rate.""")
    parser.add_argument('--warm_up', default=10, type=int, help="""Learning rate warm-up end.""")
    parser.add_argument('--lambda1', default=0.1, type=float, help="""Balancing hyper-paramter on self knowledge distillation.""")
    parser.add_argument('--lambda2', default=0.1, type=float, help="""Balancing hyper-paramter on bce quantization.""")
    parser.add_argument('--std', default=0.5, type=float, help="""Gaussian estimator standrad deviation.""")
    parser.add_argument('--temp', default=0.2, type=float, help="""Temperature scaling parameter on hash proxy loss.""")
    parser.add_argument('--transformation_scale', default=0.5, type=float, help="""Transformation scaling for self teacher: AlexNet=0.2, else=0.5.""")
    parser.add_argument('--alpha', default=0.05, type=float, help="""Balancing hyper-paramter on ProxyHashingLoss.""")
    parser.add_argument('--beta', default=0.1, type=float, help="""Balancing hyper-paramter on ProxyHashingLoss.""")

    parser.add_argument('--max_epoch', default=80, type=int, help="""Number of epochs to train.""")
    parser.add_argument('--eval_epoch', default=10, type=int, help="""Compute mAP for Every N-th epoch.""")
    parser.add_argument('--eval_init', default=50, type=int, help="""Compute mAP after N-th epoch.""")
    parser.add_argument('--output_dir', default=".", type=str, help="""Path to save logs and checkpoints.""")
    parser.add_argument('--test', default=1, type=str, help="""Number of test.""")

    return parser

class HashingEncoder(nn.Module):
    """
    hashing encoder, linear projection & tach.
    """

    def __init__(self, org_dim, k_bits):
        super().__init__()
        self.Hash = nn.Sequential(
            nn.Linear(org_dim, k_bits),
            nn.LayerNorm(k_bits)
        )

    def forward(self, x):
        return torch.tanh(self.Hash(x))
    
class PnetEncoder(nn.Module):
    def __init__(self, org_dim, k_bits):
        super().__init__()
        self.Pnet = nn.Sequential(
            nn.Linear(org_dim, k_bits),
        )

    def forward(self, x):
        x = self.Pnet(x)
        return torch.tanh(x)

class Hash_func(nn.Module):
    def __init__(self, fc_dim, N_bits, NB_CLS):
        super(Hash_func, self).__init__()
        self.bits_list = dict()
        self.P = dict()
        self.N_bits = N_bits
        self.Pnet = nn.ModuleList(
            PnetEncoder(NB_CLS, one)
            for one in self.N_bits
        )
        self.Hash = nn.ModuleList(
            HashingEncoder(fc_dim, one)
            for one in self.N_bits
        )

    def forward(self, X, P):
        for i, one in enumerate(self.N_bits):
            self.bits_list[str(one)] = self.Hash[i](X)
            self.P[str(one)] = self.Pnet[i](P)
        return self.bits_list, self.P

def train(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = T.device('cuda')
    # visualization
    vis_env = 'main'
    vis_port = 8888  # visdom port
    flag = args.dataset
    sk_bits_list = list(map(int, args.N_bits.split(",")))  # str -> list

    vis_env = '{}_{}_full_{}'.format(flag, args.N_bits,
                                     'test23_5_2_4_5_pl2_{}_batch{}_l1{}_l2{}_xiuzheng4_savemodel'.format(args.init_lr, args.batch_size, args.lambda1, args.lambda2))
    vis = Visualizer(vis_env, port=vis_port, server="http://10.109.119.102")

    path = args.data_dir
    dname = args.dataset

    accumulation_steps = args.accumulation_steps

    N_bits = args.N_bits
    init_lr = args.init_lr
    batch_size = args.batch_size


    if dname=='imagenet':
        NB_CLS=100
        Top_N=1000
        
    elif dname=='nuswide':
        NB_CLS=21
        Top_N=5000
    elif dname=='nuswide_m':
        NB_CLS=21
        Top_N=5000
    elif dname=='coco':
        NB_CLS=80
        Top_N=5000
    elif dname=='cifar-10':
        NB_CLS=10
        Top_N=54000
    else:
        print("Wrong dataset name.")
        return

    if dname != 'cifar-10':
        Img_dir = '/home/admin/code/Hash/VTS/VisionTransformerHashing-main/datasets/'+dname+'/'+dname+'256'
        Train_dir = '/home/admin/code/Hash/VTS/VisionTransformerHashing-main/datasets/'+dname+'/'+dname+'_Train.txt'
        Gallery_dir = '/home/admin/code/Hash/VTS/VisionTransformerHashing-main/datasets/'+dname+'/'+dname+'_DB.txt'
        Query_dir = '/home/admin/code/Hash/VTS/VisionTransformerHashing-main/datasets/'+dname+'/'+dname+'_Query.txt'

    if dname != 'cifar-10':
        org_size = 256
        input_size = 224
    else:
        org_size = 224
        input_size = 224

    AugT = Augmentation(org_size, args.transformation_scale)

    if dname != 'cifar-10':
        Crop = nn.Sequential(Kg.CenterCrop(input_size))
    else:
        Crop = nn.Sequential(nn.Identity())

    Norm = nn.Sequential(Kg.Normalize(mean=T.as_tensor([0.485, 0.456, 0.406]), std=T.as_tensor([0.229, 0.224, 0.225])))

    if dname != 'cifar-10':
        trainset = Loader(Img_dir, Train_dir, NB_CLS)
        trainloader = T.utils.data.DataLoader(trainset, batch_size=batch_size, drop_last=True,
                                                shuffle=True, num_workers=args.num_workers)
    else:
         trainloader, testloader, databaseloader = cifar_dataset(batch_size=batch_size, num_workers=args.num_workers)

    if args.encoder=='AlexNet':
        Baseline = AlexNet()
        fc_dim = 4096
    elif args.encoder=='ResNet':
        Baseline = ResNet()
        fc_dim = 2048
    elif args.encoder=='ViT':
        Baseline = ViT('vit_base_patch16_224')
        fc_dim = 768
    elif args.encoder=='DeiT':
        Baseline = DeiT('deit_base_distilled_patch16_224')
        fc_dim = 768
    elif args.encoder=='SwinT':
        Baseline = SwinT('swin_small_patch4_window7_224.ms_in22k_ft_in1k')
        fc_dim = 2048
    else:
        print("Wrong dataset name.")
        return

    H = Hash_func(fc_dim, sk_bits_list, NB_CLS)
    
    Baseline.cuda(device)
    H.cuda(device)

    HP_criterion = HashProxy(args.temp)
    PH_criterion = ProxyHashingLoss(args)
    REG_criterion = BCEQuantization(args.std)

    Y = torch.eye(NB_CLS, NB_CLS).cuda()

    params = [{'params': Baseline.parameters(), 'lr': 0.05*init_lr},
            {'params': H.parameters()}]

    optimizer = T.optim.Adam(params, lr=init_lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(trainloader), eta_min=0, last_epoch=-1)

    MAX_mAP = dict()
    for one in sk_bits_list:
        MAX_mAP[str(one)] = 0.0
    C_loss_list = []
    S_loss_list = []
    R_loss_list = []
    lr_baseline_list = []
    lr_H_list = []

    for epoch in range(args.max_epoch):  # loop over the dataset multiple times
        print('Epoch:', epoch, 'LR:', optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'])
        C_loss = 0.0 #哈希代理损失
        S_loss = 0.0 
        R_loss = 0.0 #量化损失

        for i, data in tqdm(enumerate(trainloader)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            l1 = T.tensor(0., device=device)
            l2 = T.tensor(0., device=device)
            l3 = T.tensor(0., device=device)

            It = Norm(Crop(AugT(inputs)))

            Xt = Baseline(It)
            Hcodes, P = H(Xt, Y)
            for one in sk_bits_list:
                l1 += HP_criterion(Hcodes[str(one)], P[str(one)], labels)
                l2 += PH_criterion(P[str(one)], labels) * args.lambda1
                l3 += REG_criterion(Hcodes[str(one)]) * args.lambda2

            loss = l1 + l2 + l3

            loss = loss / accumulation_steps

            loss.backward()
            optimizer.step()
            C_loss += l1.item()
            S_loss += l2.item()
            R_loss += l3.item()

        save__dir = "./data"
        if True:  # and (epoch+1) >= args.eval_init:
            mAPs, query_c, gallery_c, query_y, gallery_y = DoRetrieval_cifar3(device, Baseline.eval(), H.eval(), Y, databaseloader, testloader, NB_CLS, Top_N, args)
            for one in sk_bits_list:
                if mAPs[str(one)] > MAX_mAP[str(one)]:
                    MAX_mAP[str(one)] = mAPs[str(one)]
                    save_mat(query_c[str(one)], gallery_c[str(one)], query_y, gallery_y, save__dir, output_dim=one, map=MAX_mAP[str(one)], dname=args.dataset, test=args.test, l1=args.lambda1, l2=args.lambda2, mode_name="_i2i")
                    checkpoint = {
                        'map': MAX_mAP[str(one)],
                        'epoch': epoch + 1,
                        'base_model': Baseline.state_dict(),
                        'hash_model': H.state_dict(),
                        }
                    dir_name = 'data/' + str(args.dataset) + '/' + 'models/' + str(one) + 'bit_' + str(epoch) + 'e_' + str("{:.4g}".format(MAX_mAP[str(one)])) + '.pth'
                    torch.save(checkpoint, dir_name)

        if epoch >= args.warm_up:
            scheduler.step()

        Baseline.train()
        H.train()

        C_loss_list.append(C_loss / (i + 1))
        S_loss_list.append(S_loss / (i + 1))
        R_loss_list.append(R_loss / (i + 1))
        lr_baseline_list.append(optimizer.param_groups[0]['lr'])
        lr_H_list.append(optimizer.param_groups[1]['lr'])
        print('...epoch: %3d, C_loss: %3.3f' % (epoch + 1, C_loss_list[-1]))
        print('...epoch: %3d, S_loss: %3.3f' % (epoch + 1, S_loss_list[-1]))
        print('...epoch: %3d, R_loss: %3.3f' % (epoch + 1, R_loss_list[-1]))
        print('...epoch: %3d, total_loss: %3.3f' % (epoch + 1, C_loss_list[-1] + S_loss_list[-1] + R_loss_list[-1]))
        for one in sk_bits_list:
            print('...epoch: {}, mAP_{}: {}, MAX mAP_{}: {}'.format(epoch + 1, str(one), mAPs[str(one)], str(one), MAX_mAP[str(one)]))
        vis.plot('C_loss', C_loss_list[-1])
        vis.plot('S_loss', S_loss_list[-1])
        vis.plot('R_loss', R_loss_list[-1])
        vis.plot('total_loss', C_loss_list[-1] + S_loss_list[-1] + R_loss_list[-1])
        vis.plot('lr_baseline', lr_baseline_list[-1])
        vis.plot('lr_H', lr_H_list[-1])
        for one in sk_bits_list:
            vis.plot('mAP_{}'.format(one), mAPs[str(one)].item())
            vis.plot('MAX_mAP_{}'.format(one), MAX_mAP[str(one)].item())
        vis.save()

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser('DHD', parents=[get_args_parser()])
    args = parser.parse_args()
    train(args)
