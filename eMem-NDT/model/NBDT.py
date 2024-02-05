import torch
import torch.nn.functional as F


class  design_tree():

    def __init__(self,value,meaning=None,number=0):

        self.value=value
        self.meaning=meaning
        self.children=[]
        self.father=None
        self.path_problity=torch.ones_like(value).to('cuda:0')
        self.decision_probility=torch.ones_like(value).to('cuda:0')
        self.number=number

class nbdt():
    def __init__(self,net,constrcut_condition:list,leaf_dict:list,):
        self.net=net
        self.constrcut_condition=constrcut_condition
        self.leaf_dict=leaf_dict

    def __call__(self, x, n, a):
        value = self.net(x, n, a)
        node_list = list(range(0, value.shape[1] + len(self.constrcut_condition)))

        # 初始化叶子节点
        pointer = 0
        while pointer < value.shape[1]:
            node_list[pointer] = design_tree(value[:, pointer].unsqueeze(1), meaning=self.leaf_dict[pointer],
                                             number=pointer)
            pointer += 1
        del pointer

        # 创建父节点
        for j in range(0, len(self.constrcut_condition)):
            values = torch.zeros((value.shape[0], 1)).to('cuda:0')
            k = 0
            temper_father = design_tree(value=torch.zeros((value.shape[0], 1)))

            while k < len(self.constrcut_condition[j]) - 1:
                values += node_list[int(self.constrcut_condition[j][k])].value
                temper_father.children.append(node_list[int(self.constrcut_condition[j][k])])
                k += 1
            if k == 0:
                raise ValueError("Division by zero: the length of constrcut_condition[j] is less than or equal to 1.")
            father_value = values / k
            temper_father.value = father_value
            temper_father.meaning = self.constrcut_condition[j][k]
            temper_father.number = j + value.shape[1]
            node_list[j + value.shape[1]] = temper_father

            del values, temper_father, k

        # 连接子节点和父节点
        reverse_point = len(node_list) - 1
        while reverse_point > value.shape[1] - 1:
            for w in range(0, len(node_list[reverse_point].children)):
                node_list[reverse_point].children[w].father = node_list[reverse_point]
            reverse_point -= 1
        del reverse_point

        # 前向传播
        reverse_point = len(node_list) - 1
        while reverse_point > value.shape[1] - 1:
            temper_value = node_list[reverse_point].children[0].value
            for w in range(1, len(node_list[reverse_point].children)):
                temper_value = torch.concat((temper_value, node_list[reverse_point].children[w].value), dim=1)
            probility = F.softmax(temper_value, dim=1)

            for q in range(0, probility.shape[1]):
                node_list[reverse_point].children[q].decision_probility = probility[:, q].unsqueeze(1)
                node_list[reverse_point].children[q].path_problity = probility[:, q].unsqueeze(1) * node_list[
                    reverse_point].path_problity

            reverse_point -= 1
            del temper_value, probility

        out_1 = node_list[0].path_problity
        for i in range(1, value.shape[1]):
            out_1 = torch.concat((out_1, node_list[i].path_problity), dim=1)
        del node_list
        del value

        return out_1

    def train(self,data,epoch,tree_loss_value,original_loss=torch.nn.CrossEntropyLoss(),lr=0.0003,tree_noly=False):
        print("train begin\n")
        opti=torch.optim.Adam(self.net.parameters(),lr=lr)
        tree_loss=torch.nn.NLLLoss()
        for i in range(1,epoch+1):
            loss_batch=0
            batch_count=0#记录一个epoch一共多少个样本
            win = 0  # 记录识别正确（每一个epoch）
            for x,y in data:
                x = x.to(torch.float32)
                y = y.to(torch.long)
                y_hat2,nodelist,y_hat1=self.__call__(x)
                for w in range(0, y_hat2.shape[0]):
                    if torch.argmax(y_hat2[w]).item() == y[w].item():
                        win += 1
                if not tree_noly:
                    loss_original=original_loss(y_hat1,y)
                    loss_tree=tree_loss(torch.log(y_hat2),y)
                    final_loss=loss_original+tree_loss_value*loss_tree
                else:
                    final_loss=tree_loss(torch.log(y_hat2),y)
                batch_count+=x.shape[0]
                loss_batch+=final_loss
                opti.zero_grad()
                final_loss.backward()
                opti.step()
            print('epoch:', i , 'epoch_loss:', loss_batch)
        torch.save(self.net,'tree_net_weight4.pt')
    def net_alculate(self ,value):

        return self.net(value)



    def show_decision(self,x):

        decision_list={}
        y_hat2, node_list, y_hat1 = self.__call__(x)
        start_point=int(torch.argmax(y_hat2).item())##得到某一个子节点的编号
        decision_list[node_list[start_point].meaning]=node_list[start_point].decision_probility
        pointer=node_list[start_point].father.number
        while pointer!=node_list[-1].number:
            decision_list[node_list[pointer].meaning]=node_list[pointer].decision_probility
            pointer = node_list[pointer].father.number

        print(decision_list)
        return decision_list












