from sklearn.metrics import classification_report
import blvd_data
import torch
import egcn_o
import transformer_based2
import time
import numpy as np
from collections import OrderedDict
import emeNDT
import Loki_data
import MCNN_BIlstm
import transformer
import  torch.nn
np.random.seed(42)
torch.cuda.manual_seed(42)
torch.manual_seed(42)


def train_net(loss,device,data,epoch,optimize,net):
    for i in range(1,epoch+1):
        epoch_loss=0
        epoch_count=0
        win=0
        dataloder=blvd_data.DataLoader(batch_size=64)
        for t,n,a,image,y,t_f in dataloder(data):
            t_transforemer_in=t[:,:,2:].to(torch.float32).to(device)
            y=torch.tensor(y).to(device).to(torch.long)
            y_hat=net(t_transforemer_in)
            batch_loss=loss(y_hat,y)
            epoch_loss+=float(batch_loss)
            for k in range(y_hat.shape[0]):
                if torch.argmax(y_hat[k]).item()==y[k].item():
                    win+=1
            epoch_count+=y_hat.shape[0]
            batch_loss.backward()
            optimize.step()
            optimize.zero_grad()
        print('epoch_loss:',epoch_loss,'acc:',win/epoch_count,'win:',win)
def test_net(net,loss,data,dataoder):
    test_loss = 0
    total = 0
    y_pred_list = []
    y_true_list = []
    test_data = dataoder(data)

    for t, n, a, image, y,t_f in test_data:
        t_transforemer_in = t[:, :, 2:].to(torch.float32).to(device)
        y = y.to(device).to(torch.long)

        y_pred = torch.argmax(net(t_transforemer_in), dim=1)
        y_pred_list.extend(y_pred.cpu().numpy())
        y_true_list.extend(y.cpu().numpy())

        test_loss += loss(net(t_transforemer_in), y).item()

        total += y.size(0)

    report = classification_report(y_true_list, y_pred_list)
    print('Test Loss:', test_loss)
    print(report)
def train_loki(data_set,net,opti,epoch,loss):
    print('train_start')
    for i in range(0,epoch):
        epoch_loss=0
        win=0
        total=0
        data=Loki_data.DataLoader(batch_size=24)
        for t,n,a,y,t_f in data(data_set):
            t_transforemer_in=t[:,:,2:].to(torch.float32).to(device)
            y=torch.tensor(y).to(device).to(torch.long)
            y_hat=net(t_transforemer_in)

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


def test_loki(test_d, net, loss):
    test_loss = 0
    total = 0
    y_pred_list = []
    y_true_list = []
    test_data = test_loader(test_d)

    for t, n, a, y,t_f in test_data:
        t_transforemer_in = t[:, :, 2:].to(torch.float32).to(device)
        y = y.to(device).to(torch.long)

        y_pred = torch.argmax(net(t_transforemer_in), dim=1)
        y_pred_list.extend(y_pred.cpu().numpy())
        y_true_list.extend(y.cpu().numpy())

        test_loss += loss(net(t_transforemer_in), y).item()

        total += y.size(0)

    report = classification_report(y_true_list, y_pred_list)
    print('Test Loss:', test_loss)
    print(report)


if __name__=="__main__":

    loki=False
    time_start = time.time()
    if not loki:
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
    else:
        dataset = Loki_data.data_loki(future=5, input_size=3,
                                     data_path_root=r'/media/lotvs/Elements/pythonProject4/loki_data'
                                     , label_find=True, intention=False, label_need=[0, 1, 2, 3, 4, 5, 6, 7, 8])
        print(dataset.mapping)
        train, test_d = dataset()
        st = Loki_data.data_strong(random_rate=42, data=train)
        stat_1, _1 = st.data_statistic()
        train_oversample = st.oversample()
        stat, _ = st.data_statistic()
        train_loader = Loki_data.DataLoader(batch_size=24, drop=False)
        test_loader = Loki_data.DataLoader(batch_size=24, drop=False)
        train_data = train_loader(train_oversample)
        train_data_no_over = train_loader(train)
        test_data = test_loader(test_d)


        time_end = time.time()
        time_c = time_end - time_start
        print('time cost', time_c, 's')
        device = 'cuda:0'


    time_end = time.time()  # 结束计时
    time_c = time_end - time_start
    print('time cost', time_c, 's')
    device = 'cuda:1'
    train_d=train
    mynet=MCNN_BIlstm.my_lstm(num_classes=13)

    mynet.to(device)
    loss=torch.nn.CrossEntropyLoss()

    optimize=torch.optim.Adam(mynet.parameters(),lr=0.0008,weight_decay=0.000001)
    if not loki:
        print('BLVD_experiment_start')
        train_net(loss=loss,device=device,data=train_oversample,epoch=40  ,optimize=optimize,net=mynet)
        mynet.eval()
        test_net(net=mynet, loss=loss, data=train_d, dataoder=blvd_data.DataLoader(batch_size=64))
        test_net(net=mynet,loss=loss,data=test_d,dataoder=blvd_data.DataLoader(batch_size=64))
    else:
        print('loki_experiment start')
        train_loki(train_d,net=mynet,opti=optimize,epoch=100,loss=loss)

        mynet.eval()




        test_loki(test_d,net=mynet,loss=loss)
