
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import pandas as pd
import numpy as np

class position_encoder(nn.Module):

    def __init__(self,device:str):
        super(position_encoder, self).__init__()
        self.position=torch.jit.annotate(bool,True)
        self.device=device

    def forward(self, x:torch.Tensor):

        with torch.no_grad():
            position=torch.zeros((x.shape[1],x.shape[2]),device=self.device).to(self.device)
            dim_1=x.shape[1]
            dim_2=x.shape[2]
            for i in range(0,dim_1):
                for j in range(0,dim_2):
                    ##这里首先需要编写一个位置函数出来
                    pos_f=i/torch.pow(torch.tensor(10000),torch.tensor((j+1)/x.shape[2]))
                    if j%2==0:##如果对应的位置是偶数则采用sin
                        position[i,j]=torch.sin(pos_f)
                    else:
                        position[i,j]=torch.cos(pos_f)
            out=x+position
            out.to(self.device)
        return  out



class  my_net(nn.Module):
    def __init__(self,seq2seq=True,device='cuda:0',batch_first=True):
        """
        此类需要一个 位置编码以及positional embadding
        """
        super(my_net, self).__init__()
        self.seq2seq = seq2seq
        self.device=device
        self.input_embedding=nn.Linear(in_features=6,out_features=36)
        self.position=position_encoder(self.device)
        self.output_embedding=nn.Linear(in_features=6,out_features=36)
        self.batch_first=batch_first

        if self.seq2seq:
            self.transformer=nn.Transformer(num_encoder_layers=2,num_decoder_layers=2,batch_first=batch_first)
        else:
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=36, nhead=6,batch_first=batch_first)
            self.transformer=nn.TransformerEncoder(self.encoder_layer,num_layers=2)
        self.predictor=nn.Sequential(nn.Linear(in_features=36,out_features=20),nn.LeakyReLU(),
                                     nn.Linear(in_features=20,out_features=13,bias=False))
    def forward(self,x):
        x_input_s=F.tanh(self.input_embedding(x))
        x_input=self.position(x_input_s)
        if self.seq2seq:
            x_out=self.transformer(x_input,x_input)
        else:
            x_out=self.transformer(x_input)###推测这里出现batch_size的影响最大
        x_out_aver=torch.mean(x_out,dim=1,keepdim=False)
        out=x_out_aver
        return out
    def predict(self,x):
        feature=self.forward(x)
        out=self.predictor(feature)
        return out




