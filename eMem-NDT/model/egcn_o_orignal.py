
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import math
import torch.nn.functional as F
import transformer_based2

class EGCN(torch.nn.Module):
    def __init__(self, feates, activation, device='cpu', skipfeats=False):
        super().__init__()

        self.feates=feates
        self.device = device
        self.skipfeats = skipfeats
        self.GRCU_layers = nn.ModuleList()
        for i in range(1,len(self.feates)):

            grcu_i = GRCU(self.feates[i-1],self.feates[i],activation=activation)
            #print (i,'grcu_i', grcu_i)
            self.GRCU_layers.append(grcu_i.to(self.device))


    def forward(self,A_list, Nodes_list,nodes_mask_list=0):

        for unit in self.GRCU_layers:
            Nodes_list = unit(A_list,Nodes_list)


        out = Nodes_list


        return out


class GRCU(torch.nn.Module):
    def __init__(self,in_feature,out_featuer,activation):
        super().__init__()

        self.evolve_weights = mat_GRU_cell(in_feature=in_feature,out_feature=out_featuer)

        self.activation = activation
        self.GCN_init_weights = torch.rand(in_feature,out_featuer).to('cuda:0')
        self.reset_param(self.GCN_init_weights)

    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)
    def forward(self,A_list,node_embs_list):#,mask_list):
        GCN_weights = self.GCN_init_weights
        out_seq = []
        for t,Ahat in enumerate(A_list):
            node_embs = node_embs_list[t]

            GCN_weights = self.evolve_weights(GCN_weights)
            node_embs = self.activation(Ahat.matmul(node_embs.matmul(GCN_weights)))

            out_seq.append(node_embs)

        return out_seq

class mat_GRU_cell(torch.nn.Module):
    def __init__(self,in_feature,out_feature):
        super().__init__()
        self.update = mat_GRU_gate(in_feature,
                                   out_feature,
                                   torch.nn.Sigmoid())

        self.reset = mat_GRU_gate(in_feature,
                                  out_feature,
                                   torch.nn.Sigmoid())

        self.htilda = mat_GRU_gate(in_feature,
                                   out_feature,
                                   torch.nn.Tanh())
        
        # self.choose_topk = TopK(feats = in_feature,
        #                         k = out_feature)

    def forward(self,prev_Q):#,prev_Z,mask):
        z_topk = prev_Q

        update = self.update(z_topk,prev_Q)
        reset = self.reset(z_topk,prev_Q)

        h_cap = reset * prev_Q
        h_cap = self.htilda(z_topk, h_cap)

        new_Q = (1 - update) * prev_Q + update * h_cap

        return new_Q

        

class mat_GRU_gate(torch.nn.Module):
    def __init__(self,rows,cols,activation):
        super().__init__()
        self.activation = activation
        #the k here should be in_feats which is actually the rows
        self.W = Parameter(torch.Tensor(rows,rows))
        self.reset_param(self.W)

        self.U = Parameter(torch.Tensor(rows,rows))
        self.reset_param(self.U)

        self.bias = Parameter(torch.zeros(rows,cols))

    def reset_param(self,t):
        #Initialize based on the number of columns
        with torch.no_grad():
            stdv = 1. / math.sqrt(t.size(1))
            t.data.uniform_(-stdv,stdv)

    def forward(self,x,hidden):

        out_1=self.W.matmul(x)
        out_2=out_1+self.U.matmul(hidden)
        out_3=out_2+self.bias
        out=self.activation(out_3)

        return out


class edge_generating(torch.nn.Module):

    def __init__(self,in_feature,hidden_feature,device):
        super().__init__()

        self.liner1=nn.Linear(in_features=in_feature,out_features=hidden_feature)
        self.liner_2=nn.Linear(in_features=in_feature,out_features=hidden_feature)
        self.device=device
    def __call__(self,x):

        out=[]
        for k in range(len(x)):
            i=x[k]
            if not isinstance(i,torch.Tensor):
                i=torch.tensor(i,dtype=torch.float32).to(self.device)
            else:
                i=i.to(torch.float32).to(self.device)
            out_1=self.liner1(i)
            out_2=self.liner_2(i)
            out_3=out_1.matmul(out_2.t())
            final_out=nn.functional.softmax(out_3,dim=1)
            out.append(final_out)
        return out

class node_frame_self_attention(torch.nn.Module):

    def __init__(self,in_feature,out_feature):

        super().__init__()

        self.Q=nn.Linear(in_features=in_feature,out_features=out_feature)
        self.k=nn.Linear(in_features=in_feature,out_features=out_feature)
        self.v=nn.Linear(in_features=in_feature,out_features=out_feature)
        self.infeature=in_feature
        self.out_feature=out_feature

    def __call__(self,w,mean=False):
        out_=[]
        for x in w:

            Q=self.Q(x)
            K=self.k(x)
            v=self.v(x)

            out=nn.functional.softmax(Q.matmul(K.t())/ math.sqrt(self.out_feature),dim=1)
            attention=out.matmul(v)
            out_final=torch.mean(attention,dim=0,keepdim=True)
            out_.append(out_final)
        return out_
    def frame_attention(self,x,mean=False):

        Q = self.Q(x)
        K = self.k(x)
        v = self.v(x)

        out = nn.functional.softmax(Q.matmul(K.t()) / math.sqrt(self.out_feature), dim=1)
        attention = out.matmul(v)
        out_final = torch.mean(attention, dim=0, keepdim=True)
        return out_final
class time_information_embedding():

    def __init__(self,how='one-hot',device='cuda:0'):
        self.how=how
        self.device=device

    def one_hot(self,x):

        with torch.no_grad():
            out=[]
            one_hot_embedding=F.one_hot(torch.arange(0,5),num_classes=5).to(torch.float32)
            s=one_hot_embedding.to(self.device)

            for i in range(len(x)):
                if not isinstance(x[i],torch.Tensor):
                    w=torch.tensor(x[i],dtype=torch.float32).to(self.device)
                else:
                    w=x[i].to(torch.float32).detach().to(self.device)
                out.append(torch.cat((w,s[i].repeat(x[i].shape[0],1)),dim=1))
            return out

    def __call__(self,x):
        out=0
        if self.how=='one-hot':
            out=self.one_hot(x)
        return out


class state_former(nn.Module):
    def __init__(self,feat_gcn,skip,device,activation=nn.ReLU(),trajectory=True,hidden=48,ad_learning=True
                 ,how='one-hot',loki_=False,loki_class=9):
        super(state_former, self).__init__()

        self.device=device
        self.time_embedding=time_information_embedding(how=how,device=self.device)
        ####admatrix生成
        if how=='one-hot':
            self.dim_extend=5
        else:
            self.dim_extend=0
        if trajectory:
            self.input_feature=6+self.dim_extend
            feat_gcn[0]=6+self.dim_extend
        else:
            self.input_feature=8+self.dim_extendn
            feat_gcn[0]=8+self.dim_extend
        self.ad_matrix=edge_generating(in_feature=self.input_feature,hidden_feature=hidden,device=self.device)
        self.ad_matrix_learning=ad_learning
        self.activation=activation

        self.trajectory_processor=transformer_based2.my_net(seq2seq=False,device=self.device)
        self.egcn=EGCN(feates=feat_gcn,activation=activation,device=self.device,skipfeats=skip)
        self.node_self_attention=node_frame_self_attention(in_feature=feat_gcn[-1],out_feature=128)

        self.frame_attention=node_frame_self_attention(in_feature=128,out_feature=256)

        if not loki_:
            self.predctor=torch.nn.Linear(in_features=292,out_features=13)
        else:
            self.predctor=torch.nn.Linear(in_features=292,out_features=loki_class)
    def forward(self,t,n,a,mean_pooling=False):
        trajectory_feature, node_process_tensor= self.reutn_final_futual(t,n,a,mean_pooling)
        final_input=torch.cat((trajectory_feature,node_process_tensor),dim=1)
        final_out=self.predctor(final_input)
        return final_out
    def reutn_final_futual(self,t,n,a,meaning_pooling=True):

        trajectory_feature = self.trajectory_processor(t)
        node_process = []
        for k in range(len(n)):
            node_time = self.time_embedding(n[k])
            if self.ad_matrix_learning:
                ad = self.ad_matrix(node_time)
            else:
                ad = a[k].copy()### problem
                if not isinstance(ad, torch.Tensor):
                    for w in range(len(ad)):
                        ad[w] = torch.tensor(ad[w], dtype=torch.float32)
                else:
                    for w in range(len(ad)):
                        ad[w] = ad[w].to(torch.float32)
            egcn_out = self.egcn(ad, node_time)
            egcn_nod_attention = self.node_self_attention(egcn_out,meaning_pooling)
            egcn_node_attention_out = torch.cat(egcn_nod_attention, dim=0)
            egcn_frame_attention = self.frame_attention.frame_attention(egcn_node_attention_out,meaning_pooling)
            node_process.append(egcn_frame_attention)
        node_process_tensor = torch.cat(node_process, dim=0)
        node_process.clear()
        return trajectory_feature,node_process_tensor







