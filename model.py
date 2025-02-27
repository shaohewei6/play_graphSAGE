import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import numpy as np
import random
from sklearn.metrics import f1_score
from collections import  defaultdict
from encoder import Encoder
from aggregators import MeanAggregator
import matplotlib.pyplot as plt
class supervisedGraphSage(nn.Module):

    def __init__(self,num_classes,enc):
        super(supervisedGraphSage,self).__init__()
        self.enc=enc
        self.xent=nn.CrossEntropyLoss()
        self.weight=nn.Parameter(
            torch.FloatTensor(num_classes,enc.embed_dim)
        )
        init.xavier_uniform(self.weight)

    def forward(self,nodes):
        embeds=self.enc(nodes)
        scores=self.weight.mm(embeds)
        return scores.t()

    def loss(self,nodes,labels):
        scores=self.forward(nodes)
        return self.xent(scores,labels.squeeze())

def load_core():
    num_nodes=2708
    num_feats=1433
    feat_data=np.zeros((num_nodes,num_feats))
    labels=np.empty((num_nodes,1),dtype=np.int64)
    node_map={}
    label_map={}
    with open("F:\graphsage-simple-master\cora/cora.content") as fp:
        for i,line in enumerate(fp):
            info=line.strip().split()
            feat_data[i,:]=[float(x) for x in info[1:-1]]
            node_map[info[0]]=i
            """记录每个节点在文件中的出现的顺序"""
            if not info[-1] in label_map:
                label_map[info[-1]]=len(label_map)
                """为标签编码"""
            labels[i]=label_map[info[-1]]
            """labels存放每个节点的label，按照文件顺序"""
    adj_list=defaultdict(set)
    with open("F:\graphsage-simple-master\cora/cora.cites") as fp:
        for i,line in enumerate(fp):
            info=line.strip().split()
            node1=node_map[info[0]]
            node2=node_map[info[1]]
            adj_list[node1].add(node2)
            adj_list[node2].add(node1)
    return feat_data,labels,adj_list

def run_core():
    np.random.seed(1)
    random.seed(1)
    num_nodes=2708
    feat_data,labels,adj_list=load_core()
    features=nn.Embedding(2708,1433)
    features.weight=nn.Parameter(torch.FloatTensor(feat_data),requires_grad=False)
    """查表权重矩阵，按照节点的序号在权重矩阵的查找的行权重"""
    agg1=MeanAggregator(features,cuda=True)
    enc1=Encoder(features,1433,128,adj_list,agg1,gcn=True,cuda=False)
    """以agg1为聚合函数，聚合后降维，由1433降到128"""
    """encoder返回"""
    agg2=MeanAggregator(lambda nodes:enc1(nodes).t(),cuda=False)
    enc2=Encoder(lambda nodes:enc1(nodes).t(),enc1.embed_dim,128,
                 adj_list,agg2,base_model=enc1,gcn=True,cuda=False)
    """K=2"""
    enc1.num_sample=5
    enc2.num_sample=5

    graphsage=supervisedGraphSage(7,enc2)
    rand_indices=np.random.permutation(num_nodes)
    test=rand_indices[:1000]
    val=rand_indices[1000:1500]
    train=list(rand_indices[1500:])
    optimizer=torch.optim.SGD(filter(lambda p:p.requires_grad,graphsage.parameters()),lr=0.7)
    for batch in range(100):
         batch_nodes=train[:256]
         random.shuffle(train)
         optimizer.zero_grad()
         loss=graphsage.loss(batch_nodes,
                             Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
         loss.backward()
         optimizer.step()
         print(batch,loss)
    val_output=graphsage.forward(val)
    trains=list(val_output.data.numpy().argmax(axis=1))
    print("F1",f1_score(labels[val],val_output.data.numpy().argmax(axis=1),average="micro"))
    trainss=[trains.count(i) for i in range(7)]
    tures=list(labels[val].squeeze())
    turess=[tures.count(i) for i in range(7)]
    plt.plot(trainss,label='predict')
    plt.plot(turess,label='true')
    plt.legend()
    plt.show()

if __name__=="__main__":
    run_core()