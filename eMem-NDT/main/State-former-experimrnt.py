
from sklearn.metrics import classification_report
import blvd_data
import torch
import egcn_o_orignal
import transformer_based2
import time
import numpy as np
import random
from collections import OrderedDict

import  Loki_data
np.random.seed(42)
torch.cuda.manual_seed(42)
torch.manual_seed(42)



time_start = time.time()  
ss = blvd_data.dataset(future=10, input_size=5, trajectory=True)
train, v_data,test_d = ss()
st = blvd_data.data_strong(random_rate=42, data=train)
stat_1, _1 = st.data_statistic()
train_oversample = st.oversample()
stat, _ = st.data_statistic()
train_loader = blvd_data.DataLoader(batch_size=24, drop=False)
test_loader = blvd_data.DataLoader(batch_size=24, drop=False)
train_data = train_loader(train_oversample)
train_data_no_over = train_loader(train)
test_data = test_loader(test_d)



time_end = time.time()
time_c = time_end - time_start
print('time cost', time_c, 's')
device = 'cuda:1'
train_d=train
dataset=blvd_data.data_loki(future=5, input_size=2,data_path_root=""
                           ,label_find=True,intention=False,label_need=[0,1,2,3])
train_l,test_d_1=dataset()


######创建模型
net = egcn_o_orignal.state_former(feat_gcn=[6, 128, 256], skip=False, device=device, ad_learning=True,
                      activation=torch.nn.LeakyReLU())
# for t,n,a,l in train_data:
#     out=net(t,n,a)

net.to(device)





lr_main = 0.0005
lr_grcu = 0.005
weight_decay_main = 0.00001
weight_decay_grcu = 0.00001


parameter_groups = [
    {'params': net.parameters(), 'lr': lr_main, 'weight_decay': weight_decay_main}
]

# 动态创建 GRCU_layer 的参数组
for grcu_layer in net.egcn.GRCU_layers:
    parameter_groups.append(
        {'params': grcu_layer.evolve_weights.parameters(), 'lr': lr_grcu, 'weight_decay': weight_decay_grcu})

optimizer = torch.optim.Adam(parameter_groups)


loss=torch.nn.CrossEntropyLoss()
def train(data_set,net,opti,epoch,loss,mean=False):
    print('train_start')
    for i in range(0,epoch):
        epoch_loss=0
        win=0
        total=0
        data=blvd_data.DataLoader(batch_size=64)
        for t,n,a,image,y,t_f in data(data_set):
            t = t.to(torch.float32).to(device)
            y = y.to(torch.long).to(device)
            y_hat = net(t, n, a,mean)

            for w in range(0, y_hat.shape[0]):
                if torch.argmax(y_hat[w]).item() == y[w].item():
                    win += 1  #

            loss_batch = loss(y_hat, y)

            opti.zero_grad()
            loss_batch.backward()
            opti.step()
            epoch_loss += float(loss_batch)
            torch.cuda.empty_cache()
            total += t.shape[0]
            # if k%10==0:
            #     opti.step()
            #     opti.zero_grad()
        print('epoch:', i + 1, 'epoch_loss:', epoch_loss, 'acc', win / total)
    return 0


def test(test_d, net, loss,mean=False):
    test_loss = 0
    total = 0
    y_pred_list = []
    y_true_list = []
    test_data = test_loader(test_d)

    for t, n, a, image, y,t_f in test_data:
        t = t.to(torch.float32).to(device)
        y = y.to(torch.long).to(device)

        y_pred = torch.argmax(net(t, n, a,mean), dim=1)
        y_pred_list.extend(y_pred.cpu().numpy())
        y_true_list.extend(y.cpu().numpy())

        test_loss += loss(net(t, n, a,mean), y).item()

        total += y.size(0)

    report = classification_report(y_true_list, y_pred_list)
    print('Test Loss:', test_loss)
    print(report)

s=train(data_set=train_oversample,net=net,opti=optimizer,loss=loss,epoch=30,mean=True)
net=net.eval()



s_1=test(test_d,net=net,loss=loss,mean=True)
print('finish')
