from config import *
import torch
from einops import rearrange
import torch.nn.init as init
from modules import *
from ConvSSM import *

class CustLayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first", elementwise_affine=True):
        super(CustLayerNorm, self).__init__()
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            if self.elementwise_affine:
                x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Fusion(nn.Module):
    def __init__(self, level, channels, N_list=None) -> None:
        super(Fusion, self).__init__()
        self.level = level  # self.channels = [96, 96, 192, 384, 768]
        self.N_list = [56, 56, 28, 14, 7]
        self.dw_conv = nn.Conv2d(channels[level + 1], out_channels=channels[level + 1], groups=channels[level + 1], bias=True, kernel_size=3, padding=1)
        self.activation = nn.ReLU(True)
        self.ConvSSM = ConvSSM(hidden_dim=channels[level + 1])
        if level in [1, 2, 3]:
            self.down = nn.Sequential(
                nn.Conv2d(channels[level], channels[level + 1], kernel_size=(2, 2), stride=2),
                CustLayerNorm(channels[level + 1], eps=1e-6, data_format="channels_first"),
                LSKAttention(d_model=channels[level + 1]),
                CustLayerNorm(channels[level + 1], eps=1e-6, data_format="channels_first"),
            )  
        else:
            self.down = nn.Sequential(
                LSKAttention(d_model=channels[level + 1]),
                CustLayerNorm(channels[level + 1], eps=1e-6, data_format="channels_first"),
            )

    def forward(self, *args):
        c_down, c_up = args
        b, h, w, c = c_down.shape
        c_down = FrequencyDistributionLearner(c, h, w, 4, self.level)(c_down)
        c_up = rearrange(c_up, 'b h w c -> b c h w')
        down = self.down(c_down)
        shortcut = c_up + down
        if (self.level+1) == 4:
            x = self.ConvSSM(self.activation(self.dw_conv(shortcut)))
            shortcut = rearrange(shortcut, 'b c h w -> b h w c')
            x = shortcut + x
        else:
            x = rearrange(shortcut + self.activation(self.dw_conv(shortcut)), 'b c h w -> b h w c')
        return x

class Augmentation(nn.Module):
    def __init__(self, org_size, Aw=1.0):
        super(Augmentation, self).__init__()
        self.gk = int(org_size*0.1)
        if self.gk%2==0:
            self.gk += 1
        self.Aug = nn.Sequential(
        Kg.RandomResizedCrop(size=(org_size, org_size), p=1.0*Aw),
        Kg.RandomHorizontalFlip(p=0.5*Aw),
        Kg.ColorJitter(brightness=0.4, contrast=0.8, saturation=0.8, hue=0.2, p=0.8*Aw),
        Kg.RandomGrayscale(p=0.2*Aw),
        Kg.RandomGaussianBlur((self.gk, self.gk), (0.1, 2.0), p=0.5*Aw))

    def forward(self, x):
        return self.Aug(x)

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()        
        self.F = nn.Sequential(*list(models.alexnet(pretrained=True).features))
        self.Pool = nn.AdaptiveAvgPool2d((6,6))
        self.C = nn.Sequential(*list(models.alexnet(pretrained=True).classifier[:-1]))
    def forward(self, x):
        x = self.F(x)
        x = self.Pool(x)
        x = T.flatten(x, 1)
        x = self.C(x)
        return x

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.pretrained = models.resnet50(pretrained=True)
        self.children_list = []
        for n,c in self.pretrained.named_children():
            self.children_list.append(c)
            if n == 'avgpool':
                break

        self.net = nn.Sequential(*self.children_list)
        self.pretrained = None
        
    def forward(self,x):
        x = self.net(x)
        x = T.flatten(x, 1)
        return x

class ViT(nn.Module):
    def __init__(self, pretrained_name):
        super().__init__()
        self.pm = timm.create_model(pretrained_name, pretrained=True)
    def forward(self, x):
        x = self.pm.patch_embed(x)
        cls_token = self.pm.cls_token.expand(x.shape[0], -1, -1)
        x = T.cat((cls_token, x), dim=1)
        x = self.pm.pos_drop(x + self.pm.pos_embed)
        x = self.pm.blocks(x)
        x = self.pm.norm(x)
        return x[:, 0]

class DeiT(nn.Module):
    def __init__(self, pretrained_name):
        super().__init__()
        self.pm = timm.create_model(pretrained_name, pretrained=True)
    def forward(self, x):
        x = self.pm.patch_embed(x)
        cls_token = self.pm.cls_token.expand(x.shape[0], -1, -1)
        x = T.cat((cls_token, self.pm.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pm.pos_drop(x + self.pm.pos_embed)
        x = self.pm.blocks(x)
        x = self.pm.norm(x)
        return x[:, 0]

class SwinT(nn.Module):
    def __init__(self, pretrained_name, hidden_dim=2048):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pm = timm.create_model(pretrained_name, pretrained=True, local_only=True, num_classes=self.hidden_dim)
        self.channels = [96, 96, 192, 384, 768]
        self.fusion1 = Fusion(level=0, channels=self.channels)
        self.fusion2 = Fusion(level=1, channels=self.channels)
        self.fusion3 = Fusion(level=2, channels=self.channels)
        self.fusion4 = Fusion(level=3, channels=self.channels)
        # 多尺度融合
        # 获取swinT各stage
        self.patch_embd = self.pm.patch_embed
        self.stage1 = self.pm.layers[0]
        self.stage2 = self.pm.layers[1]
        self.stage3 = self.pm.layers[2]
        self.stage4 = self.pm.layers[3]
        
        self.norm_layer = self.pm.norm
        self.class_layer = self.pm.head

        self.ae = nn.Sequential(
            nn.Linear(in_features=self.hidden_dim, out_features=8192),
            nn.LayerNorm(8192),
            nn.ReLU(True),
            nn.Linear(in_features=8192, out_features=self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(True)
        )

        self.weight_init()

    def weight_init(self):
        initializer = self.kaiming_init
        for m in self.ae:
            initializer(m)

    def kaiming_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            m.weight.data.fill_(1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        patch_embed_out = self.patch_embd(x.to(x.device, non_blocking=True))  # torch.Size([10, 56, 56, 96])

        stage1_out = self.stage1(patch_embed_out)  # torch.Size([10, 56, 56, 96])
        fusion_1 = self.fusion1(patch_embed_out, stage1_out)  # torch.Size([10, 56, 56, 96])
        fusion_1_cat_stage1_out = fusion_1 + stage1_out  # torch.Size([10, 56, 56, 96])

        stage2_out = self.stage2(fusion_1_cat_stage1_out)  # torch.Size([10, 28, 28, 192])
        fusion_2 = self.fusion2(fusion_1_cat_stage1_out, stage2_out)  # torch.Size([10, 28, 28, 192])
        fusion_2_cat_stage2_out = fusion_2 + stage2_out  # torch.Size([10, 28, 28, 192])

        stage3_out = self.stage3(fusion_2_cat_stage2_out)  # torch.Size([10, 14, 14, 384])
        fusion_3 = self.fusion3(fusion_2_cat_stage2_out, stage3_out)  # torch.Size([10, 14, 14, 384])
        fusion_3_cat_stage3_out = fusion_3 + stage3_out  # torch.Size([10, 14, 14, 384])

        stage4_out = self.stage4(fusion_3_cat_stage3_out)  # torch.Size([10, 7, 7, 768])
        fusion_4 = self.fusion4(fusion_3_cat_stage3_out, stage4_out)  # torch.Size([10, 7, 7, 768])
        fusion_4_cat_stage4_out = fusion_4 + stage4_out  # torch.Size([10, 7, 7, 768])

        norm_out = self.norm_layer(fusion_4_cat_stage4_out)
        fe = self.class_layer(norm_out).squeeze()

        fe = self.ae(fe)

        return fe


if __name__ == "__main__":
    pass
