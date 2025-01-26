from config import *
import torch.nn.functional as F
import torch
import torch.nn as nn


class HashProxy(nn.Module):
    def __init__(self, temp):
        super(HashProxy, self).__init__()
        self.temp = temp

    def forward(self, X, P, L):

        X = F.normalize(X, p = 2, dim = -1)
        P = F.normalize(P, p = 2, dim = -1)
        
        D = F.linear(X, P) / self.temp
        # print(L)
        # print(L.shape)
        L /= T.sum(L, dim=1, keepdim=True).expand_as(L)

        xent_loss = T.mean(T.sum(-L * F.log_softmax(D, -1), -1))
        return xent_loss

class ProxyHashingLoss(nn.Module):
    def __init__(self, args):
        super(ProxyHashingLoss, self).__init__()
        self.alpha = args.alpha
        self.beta = args.beta
        self.k = args.N_bits

    def forward(self, G_hat, L):
        """
        G_hat: 代理哈希码矩阵，形状为(c, k)，其中c是类别数，k是哈希码长度
        """
        # c, k = G_hat.shape
        
        # 计算a_i
        a_i = G_hat.sum(dim=0)  # 按列求和，形状为(k,)
        
        # 计算G_hat与其转置的矩阵乘法，得到形状为(c, c)的相似度矩阵
        similarity_matrix = torch.matmul(G_hat, G_hat.t())

        # print(L.shape)
        L_s = torch.matmul(L.T , L).sum(dim=-1)
        L_p = L_s / L_s.sum()
    
        # 将相似度矩阵的对角线元素设置为0，因为我们不考虑自身的相似度
        # eye = torch.eye(G_hat.size(0), device=G_hat.device)
        # similarity_matrix = similarity_matrix * (1 - eye)
        
        # 使用ReLU函数保留正相似度，忽略负相似度
        positive_similarity = torch.relu(similarity_matrix)

        # 计算总的相似性损失
        similarity_loss = positive_similarity.sum(dim=-1)
        similarity_loss = torch.matmul(L_p.T, similarity_loss)


        # 计算第二部分损失：代理哈希码的正则化项
        regularization_loss = self.alpha * torch.norm(a_i, p='fro') ** 2
        
        # 计算第三部分损失：代理哈希码与其符号函数之间的量化损失
        quantization_loss = self.beta * torch.norm(G_hat - torch.sign(G_hat), p='fro') ** 2
        
        # 总损失
        total_loss = similarity_loss + regularization_loss + quantization_loss
        
        return total_loss

class BCEQuantization(nn.Module):
    def __init__(self, std):
        super(BCEQuantization, self).__init__()
        # self.BCE = BCEFocalLosswithLogits()
        self.BCE = nn.BCELoss()
        self.std=std
    def normal_dist(self, x, mean, std):
        prob = T.exp(-0.5*((x-mean)/std)**2)
        return prob
    def forward(self, x):
        x_a = self.normal_dist(x, mean=1.0, std=self.std)
        x_b = self.normal_dist(x, mean=-1.0, std=self.std)
        y = (x.sign().detach() + 1.0) / 2.0
        l_a = self.BCE(x_a, y)
        l_b = self.BCE(x_b, 1-y)
        return (l_a + l_b)
