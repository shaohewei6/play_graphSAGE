import torch
import torch.nn as nn
from torch.autograd import Variable
import random

class MeanAggregator(nn.Module):
    def __init__(self,features,cuda=False,gcn=False):
        super(MeanAggregator,self).__init__()

        self.features=features
        """feature是一个函数"""
        self.cuda=cuda
        self.gcn=gcn

    def forward(self,nodes,to_neighs,num_sample=10):
        _set=set
        if not num_sample is None:
            _sample=random.sample
            """随机采样（采样集合，采样数）"""
            sample_neighs=[_set(_sample(to_neigh,num_sample))
                           if len(to_neigh)>=num_sample else to_neigh for to_neigh in to_neighs]
            """当邻居节点的数目大于采样数时，进行采样，如果不大于则直接拿邻居节点集合作为采样集合"""
        else:
            sample_neighs=to_neighs

        if self.gcn:
            sample_neighs=[sample_neigh+_set([nodes[i]]) for i,sample_neigh in enumerate(sample_neighs)]
            """卷积加上自身节点"""
        unique_nodes_list=list(set.union(*sample_neighs))
        """删除重复节点，并合并成一个列表,代表采样的总的节点的列表"""
        unique_nodes={n:i for i,n in enumerate(unique_nodes_list)}
        """为节点编个号"""
        mask=Variable(torch.zeros(len(sample_neighs),len(unique_nodes)))
        """构建掩码二维张量"""
        column_indices=[unique_nodes[n] for sample_neigh in sample_neighs for n in sample_neigh]
        row_indices=[i for i in range(len(sample_neighs)) for j in range(len(sample_neighs[i]))]
        """j用作计数器，即sample_neigh中有几个节点就有几个i"""
        mask[row_indices,column_indices]=1
        if self.cuda:
            mask.cuda()
        num_neigh=mask.sum(1,keepdim=True)
        """记录每次采样的个数"""
        mask=mask.div(num_neigh)
        if self.cuda:
            embed_matrix=self.features(torch.LongTensor(unique_nodes_list).cuda())
        else:
            embed_matrix=self.features(torch.LongTensor(unique_nodes_list))
        to_feats=mask.mm(embed_matrix)
        """相当于相加再取平均"""
        """sample_numbers(节点的个数) * embed_dim"""
        return to_feats