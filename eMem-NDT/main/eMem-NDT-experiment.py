import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from sklearn.metrics import classification_report
import blvd_data
import torch
import egcn_o_orignal
import transformer_based2
import time
import numpy as np
from collections import OrderedDict
import eMem_NDT
import Loki_data
np.random.seed(42)
torch.cuda.manual_seed(42)
torch.manual_seed(42)



time_start = time.time()
ss = blvd_data.dataset(future=10, input_size=5, trajectory=True,data_path_root=r'D:\pythonProject4\BLVD\data')
train_da,test_d = ss()
st = blvd_data.data_strong(random_rate=42, data=train_da)
stat_1, _1 = st.data_statistic()
train_oversample = st.oversample()
stat, _ = st.data_statistic()
train_loader = blvd_data.DataLoader(batch_size=256, drop=False)
test_loader = blvd_data.DataLoader(batch_size=24, drop=False)
train_data = train_loader(train_oversample)
train_data_no_over = train_loader(train_da)
test_data = test_loader(test_d)
print('提取完成')
print(1)
time_end = time.time()
time_c = time_end - time_start
print('time cost', time_c, 's')
device = 'cuda:0'
train_d=train_da
# dataset=Loki_data.data_loki(future=10, input_size=5,data_path_root=r'/media/lotvs/Elements/pythonProject4/loki_data'
#                            ,label_find=True,intention=False,label_need=[7,8])
# train_l,test_d_1=dataset()
# train_loader_1 = Loki_data.DataLoader(batch_size=6, drop=False)


net = egcn_o_orignal.state_former(feat_gcn=[6, 128, 256], skip=False, device=device, ad_learning=True,
                      activation=torch.nn.LeakyReLU())
# for t,n,a,l in train_data:
#     out=net(t,n,a)

net.to(device)
construct_condition_2_blvd = [[0, 1, 'overtaking maneuvers'], [9, 10, 'parallel driving'],
                              [5, 6, 'merging into traffic']
    , [3, 4, 'diverting behavior'], [2, 7, 'speed adjustment'], [15, 16, 'lane change behavior'],
                              [14, 18, 'lane interaction behavior'], [13, 19, 'active driving behavior'],
                              [8, 17, 'straight driving behavior'], [11, 20, 'reactive driving behavior'],
                              [21, 22, 'driving strategy'], [12, 23, 'driving behavior']]## example of the tree
leaf_node_information = ['overtaking_from_left', 'overtaking_from_right', 'decelerate_straight', 'driving_away_to_left',
                         'driving away to right', 'driving in from right', 'driving in from left',
                         'accelerate straight',
                         'uniformly straight driving', 'parallel driving in left', 'parallel driving in right',
                         'stopping', 'others']




tree = eMem_NDT.memory_tree(construction_condition=construct_condition_2_blvd,
                                leaf_imformation=leaf_node_information,
                                memory_dataset=train_data_no_over, filter_value=0.8, memory_all=False, random_mem=False
                                , mean=False, logic=True, net=net, device=device, consistency=False,
                                same_tansfer=False, load=False, random_number=300,final=True)

print('construction start')
tree.tree_construction()
tree.show_memory_number()
# train_data_2 = train_loader(train_oversample)
loss = torch.nn.CrossEntropyLoss()
net2=eMem_NDT.normal(mean=False)
net2.to(device)
parameter_groups_0 = []
for i in tree.node_list[0:13]:
    parameter_groups_0.append({'params': i.net.parameters(), 'lr': 0.00005, 'weight_decay': 0.000001})
parameter_groups_0.append({'params':net2.parameters(),'lr':0.00005,'weight_decay': 0.000001})
optimizer = torch.optim.Adam(params=parameter_groups_0)






treeloss=torch.nn.NLLLoss()
def train(train_oversample,net,net2,tree,loss,opt,train_loader,epoch=5):
    final=True

    for k in range(epoch):
        epoch_loss = 0
        total = 0
        win = 0

        train_data_3=train_loader(train_oversample)
        for t, n, a, image, y,t_f in train_data_3:
            t = t.to(torch.float32).to(device)
            y = y.to(torch.long).to(device)
            if not final:
                t_, s = net.reutn_final_futual(t, n, a,False)

                y_hat = torch.cat((t_, s), dim=1)
            y_hat=net(t, n, a)
            # for i in range(t.shape[0]):
            #      tree.show_decision_soft(input=y_hat[i].unsqueeze(0),t=t[i],n=n[i],t_f=t_f[i],how='visual')
            y_hat_2=net2(y_hat)#transf
            out=tree(y_hat_2)
            loss_batch = loss(torch.log(out),y)

            epoch_loss += float(loss_batch)
            total += out.shape[0]
            for i in range(0, out.shape[0]):
                if int(torch.argmax(out[i]).item()) == int(y[i].item()):
                    win += 1

            loss_batch.backward()
            opt.step()
            opt.zero_grad()
        print('loss:', epoch_loss, 'acc:', win / total, 'total:', total, 'win:', win)




def test(test_d,test_loader,net,tree,net2,leaf_node_information ):
    final=True
    print('test_strat')
    test_loss = 0
    total = 0
    y_pred_list = []
    y_true_list = []
    test_data = test_loader(test_d)
    for t, n, a, image, y, t_f in test_data:
        t = t.to(torch.float32).to(device)
        y = y.to(torch.long).to(device)
        # t_, s = net.reutn_final_futual(t, n, a)

        if not final:
            t_, s = net.reutn_final_futual(t, n, a)

            y_hat = torch.cat((t_, s), dim=1)
        y_hat= net(t, n, a,mean_pooling=False)
        y_hat_2 = net2(y_hat)  # transf
        for i in range(t.shape[0]):
            print(leaf_node_information[y[i]])
            tree.show_decision_soft(input=y_hat_2[i].unsqueeze(0),t=t[i],n=n[i],t_f=t_f[i],how='visual')
            input("any key to continue")
        y_hat_2 = net2(y_hat)  # transf
        out = tree(y_hat_2)
        y_pred=torch.argmax(out,dim=1)
        y_pred_list.extend(y_pred.cpu().numpy())
        y_true_list.extend(y.cpu().numpy())

        test_loss += loss(out, y).item()

        total += y.size(0)

    report = classification_report(y_true_list, y_pred_list)
    print('Test Loss:', test_loss)
    print(report)
    index_statstical=[]
    for i in range(len(tree.leaf_dic)):
        index_statstical.append(tree.node_list[i].memory_statistical())
    print (index_statstical)

train(train_oversample=train_oversample,net=net,net2=net2,tree=tree,loss=treeloss,opt=optimizer,epoch=5)
tree=tree.eval()
ent=net.eval()
net_2=net2.eval()



test(test_d=test_d,test_loader=test_loader,net=net,tree=tree,net2=net2,leaf_node_information=leaf_node_information )

