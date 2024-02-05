import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import random
import transformer_based2
import random
import matplotlib.pyplot as plt

from collections import Counter
from matplotlib.lines import Line2D




class normal(nn.Module):
    def __init__(self, mean):
        super(normal, self).__init__()
        self.net = nn.Sequential(nn.Linear(13, 300), nn.ReLU(), nn.Linear(300, 600), nn.LeakyReLU(),
                                 nn.Linear(600, 100), nn.ReLU()
                                 , nn.Linear(100, 13), nn.LeakyReLU())
        self.mean = mean

    def __call__(self, x):
        out = self.net(x)
        if self.mean:
            out = torch.mean(out, dim=0)
        return out

"""
class node_transformer(nn.Module):
    def __init__(self, mean):
        super(node_transformer, self).__init__()
        self.input_embedding = nn.Linear(in_features=13, out_features=256)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.projection = nn.Sequential(nn.Linear(256, 200), nn.LeakyReLU(), nn.LeakyReLU()
                                        , nn.Linear(200, 13), nn.LeakyReLU())
        self.mean = mean

    def __call__(self, x):
        x = F.leaky_relu(self.input_embedding(x))
        out_1 = self.transformer(x)
        out_2 = self.projection(out_1)
        if self.mean:
            out_2 = torch.mean(out_2, dim=0)
        return out_2


class attention_node(nn.Module):
 

    def __init__(self, hidden_dim, out_dim, mean=True):
 
        super(attention_node, self).__init__()
        self.qkv_in = hidden_dim
        self.qkv_out = hidden_dim + 40
        ### 下面分别创建qkv对应的映射层
        self.q = nn.Linear(in_features=self.qkv_in, out_features=self.qkv_out)
        self.k = nn.Linear(in_features=self.qkv_in, out_features=self.qkv_out)
        self.v = nn.Linear(in_features=self.qkv_in, out_features=self.qkv_out)
        self.projection = nn.Linear(in_features=self.qkv_out, out_features=out_dim)

    def __call__(self, data):
        data_q = F.leaky_relu(self.q(data))
        data_k = F.leaky_relu(self.k(data)).permute(0, 2, 1) 
        data_v = F.leaky_relu(self.v(data))  

        alph = torch.matmul(data_q, data_k)
        alph = F.softmax(alph, dim=2)
        out = F.leaky_relu(torch.matmul(alph, data_v))
        if out.shape[0] > 1:
            out = torch.mean(out, dim=0, keepdim=False)  # 输出为attention之后的平均

        final_out = self.projection(out)
        return final_out

"""
class memory_leaf_node():


    def __init__(self, filter_value, mean, keep_input=False,
                 net=None, device='cuda:0', consistency=False, net_2=None, same_tansfer=False, random_number=400
                 , distance=False,mem_all=False):
        self.father = None
        self.meaning = 'None'  # This variable is used to store the meaning of the nodes
        if mem_all:
            self.memory = []  # Used to store instances, this variable needs to become a tensor after instantiation.
        else:
            self.memory = {'extract_feature': [],
                           'raw_feature': []}
        self.node_probility = torch.tensor(1, dtype=torch.float32).to(device)
        self.path_probility = torch.tensor(1, dtype=torch.float32).to(device)
        self.children = None
        self.score = None  #"Used to record similarity scores
        self.filter_value = filter_value  # This variable is used to control the size of memory
        self.same_transfer = same_tansfer
        if self.same_transfer:
            self.net = net_2.to(device)
        else:
            self.net = normal(mean)
            self.net = self.net.to(device)
        self.device = device
        self.mean = mean
        self.keep_input = keep_input
        if self.keep_input:
            self.former_net = net.to(device)
        self.consistency = consistency
        self.random_number = random_number
        self.number = None
        self.distance = distance
        self.memory_f =[]
        self.memory_count=[]# statsticial of the index
    def nod_memory_generation(self, f_i, t_i, n_i,t_f):

        if len(self.memory['extract_feature']) == 0:
            self.memory['extract_feature'].append(f_i)
            self.memory['raw_feature'].append((t_i,n_i,t_f))
        else:

            max_cos_value = -1
            for exp in self.memory['extract_feature']:

                t1 = torch.flatten(exp)
                t2 = torch.flatten(f_i)
                """Calculate the cosine similarity between two vectors"""
                cos_value = float(t1 @ t2) / (math.sqrt(float(t1 @ t1)) * math.sqrt(float(t2 @ t2)))
                if cos_value >= max_cos_value:
                    max_cos_value = cos_value
            """After exiting the loop, obtain the current 'value' and the instances in memory"""
            if max_cos_value <= self.filter_value:
                self.memory['extract_feature'].append(f_i)
                self.memory['raw_feature'].append((t_i, n_i,t_f))


    def nod_memory_generation_2(self, f_i,t_i,n_i,t_f):

        self.memory.append((f_i,t_i,n_i,t_f))
    def memory_wash(self):
        """
        Function purpose：Used to clean the memories generated by nod_memory_generation_2.
        """
        with torch.no_grad():
            select_memory = []
            for j in range(0, len(self.memory)):
                select_one = self.memory[j]
                t1 = torch.flatten(select_one)
                max_cos = -1
                for k in range(0, len(self.memory)):
                    if k == j:
                        continue
                    t2 = torch.flatten(self.memory[k])
                    """The next step is to calculate the cosine similarity between the two vectors."""
                    cos_value = float(t1 @ t2) / (math.sqrt(float(t1 @ t1)) * math.sqrt(float(t2 @ t2)))
                    if cos_value >= max_cos:
                        max_cos = cos_value
                """ After completing the remaining traversal, make a judgment."""
                if max_cos <= self.filter_value:
                    select_memory.append(select_one)
            """Finally, exit the loop."""
            self.memory = select_memory

    def memory_wash_2(self):
        with torch.no_grad():
            select_memory = []
            delete_list = []  #This list is used to record the indices that need to be deleted.
            for k in range(0, len(self.memory)):
                if self.keep_input:
                    t1 = torch.flatten(self.former_net(self.memory[k]))
                else:
                    t1 = torch.flatten(self.memory[k])
                for j in range(k, len(self.memory)):
                    if j == k:
                        continue
                    if self.keep_input:
                        t2 = torch.flatten(self.former_net(self.memory[k]))
                    if not self.keep_input:
                        t2 = torch.flatten(self.memory[j])
                    cos_value = t1 @ t2 / (math.sqrt(t1 @ t1) * math.sqrt(t2 @ t2))
                    if cos_value > self.filter_value:
                        delete_list.append(k)
                        break
            for q in range(0, len(self.memory)):
                if q in delete_list:
                    continue
                select_memory.append(self.memory[q])
            self.memory = select_memory

    def memory_wash_4(self):
        with torch.no_grad():
            select_memory = []
            select_index = []
            while len(self.memory) > 0:
                # randomly select a sample from memory
                k = random.randint(0, len(self.memory) - 1)
                if k in select_index:
                    continue
                if self.keep_input:
                    t1 = torch.flatten(self.former_net(self.memory[k]))
                else:
                    t1 = torch.flatten(self.memory[k])
                delete = False
                for j in range(len(self.memory)):
                    if j == k:
                        continue
                    if self.keep_input:
                        t2 = torch.flatten(self.former_net(self.memory[j]))
                    if not self.keep_input:
                        t2 = torch.flatten(self.memory[j])
                    cos_value = float(t1 @ t2) / (math.sqrt(float(t1 @ t1)) * math.sqrt(float(t2 @ t2)))
                    if cos_value > self.filter_value:
                        delete = True
                        break
                if not delete:
                    # add the selected sample to select_memory
                    select_memory.append(self.memory[k])
                    select_index.append(k)
                # remove the selected sample from memory
                # self.memory = self.memory[:k] + self.memory[k + 1:]
            self.memory = select_memory

    def memory_wash_3(self):

        random.seed(42)
        if len(self.memory) < self.random_number:
            self.random_number = len(self.memory) - 50
        select_memory = random.sample(self.memory, int(self.random_number))
        self.memory = select_memory

    def simulation_calculating_2(self, value):


        max_cos_value = -1
        for exp in self.memory:

            t1 = torch.flatten(exp)
            t2 = torch.flatten(value)

            cos_value = (t1 @ t2) / (math.sqrt(t1 @ t1) * math.sqrt(t2 @ t2))
            if cos_value >= max_cos_value:
                max_cos_value = cos_value
        return max_cos_value

    def simulation_calculating(self, value):

        max_cos_value = -1

        exp = self.net(self.memory_f)
        t2 = value
        if self.mean:

            cos_value = F.cosine_similarity(exp, t2, 1) * 30
            return cos_value
        else:
            j = t2[0]
            if self.distance:
                s = torch.dist(j.unsqueeze(0), exp.squeeze(1), 2).unsqueeze(0)
                for j in range(1, t2.shape[0]):
                    s = torch.cat((s, torch.dist(t2[j].unsqueeze(0), exp.squeeze(1)).unsqueeze(0)))
                s = -1 * s
            else:
                s = F.cosine_similarity(j.unsqueeze(0), exp).unsqueeze(0) * 30
                for j in range(1, t2.shape[0]):
                    s = torch.cat((s, F.cosine_similarity(t2[j].unsqueeze(0), exp).unsqueeze(0) * 30))


            max_values, _ = torch.max(s, dim=1, keepdim=False)
            values = max_values
            self.index = _
            self.memory_count.extend(self.index.tolist())

            del s
            del j
            return values
    def memory_statistical(self):
        distribution=Counter(self.memory_count)
        return distribution






class inner_node():
    def __init__(self, hidden_dim, out_dim, mean
                 , keep_input=False, net=None, device='cuda:0'
                 , consistency=False, net_2=None, same_transfer=False
                 ):

        self.meaning = 'None'
        self.memory = 'None'
        self.mean = mean
        self.same_transfer = same_transfer
        if self.same_transfer:
            self.net = net_2.to(device)
        else:
            self.net = normal(mean)
            self.net = self.net.to(device)
        self.father = None
        self.children = []
        self.node_probility = torch.tensor(1, dtype=torch.float32).to(device)
        self.path_probility = torch.tensor(1, dtype=torch.float32).to(device)
        self.score = None
        self.keep_input = keep_input
        if self.keep_input:
            self.former_net = net.to(device)
        self.device = device
        self.consistency = consistency
        self.number = None

    def memory_generation(self):


        d1 = self.children[0].memory
        for q in range(1, len(self.children)):
            d1 = torch.cat((d1, self.children[q].memory), dim=0)
        self.memory = d1

    def simulation_calculating(self, value):
        if self.keep_input:

            if self.consistency:
                abstract_memory = self.former_net(self.memory, self.device)
            else:
                abstract_memory = self.net(self.former_net(self.memory, device=self.device))

        else:
            abstract_memory = self.net(self.memory)
        t1 = abstract_memory
        t2 = torch.flatten(value, start_dim=1)

        if self.mean:
            cos_value = F.cosine_similarity(t1, t2, 1) * 23
            return cos_value
        else:

            j = t2[0]
            s = F.cosine_similarity(j.unsqueeze(0), t1.squeeze(1)).unsqueeze(0) * 23
            for j in range(1, t2.shape[0]):
                s = torch.cat((s, F.cosine_similarity(t2[j].unsqueeze(0), t1.squeeze(1)).unsqueeze(0) * 23))
            max_values, _ = torch.max(s, dim=1, keepdim=False)
            return max_values

    def simulation_calculating_mean(self, value):
        abstract_memory = torch.mean(self.memory, dim=0, keepdim=True).squeeze(1)
        t1 = abstract_memory
        t2 = torch.flatten(value, start_dim=1)
        cos_value = F.cosine_similarity(t1, t2, 1) * 16
        return cos_value



class memory_tree(nn.Module):


    def __init__(self, construction_condition, leaf_imformation, memory_dataset, filter_value, memory_all=False,
                 random_mem=False, mean=False, logic=False,
                 net=None, keep_input=False, device='cuda:0'
                 , consistency=False, same_tansfer=False, load=False, random_number=400, distance=False,
                 final=True):

        super(memory_tree, self).__init__()
        self.condition = construction_condition
        self.leaf_dic = leaf_imformation
        self.node_list = []
        self.data = memory_dataset
        self.device = device
        self.filter_value = filter_value
        self.parameters = []
        self.former_net = net.to(self.device)
        self.former_net = self.former_net.to(self.device)
        self.memory_all = memory_all
        self.random_mem = random_mem
        self.mean = mean
        self.logic = logic
        self.keep_input = keep_input
        self.consistency = consistency
        self.same_transfer = same_tansfer
        self.random_number = random_number

        if self.same_transfer:
            if load:
                self.node_net = torch.load('t0.pt').to(self.device)
            else:
                self.node_net = normal(self.mean).to(self.device)
        else:
            self.node_net = None
        self.distance = distance
        self.final=final

    def leaf_node_memory(self):


        """Firstly, extract the corresponding data for calculation and extract features."""
        for t, n, a, image,y,t_f in self.data:
            t = t.to(torch.float32).to(self.device)
            y = y.to(torch.long).to(self.device)

            if not self.final:
                with torch.no_grad():
                    t_,s= self.former_net.reutn_final_futual(t, n, a)
                f = torch.cat((t_,s), dim=1)
            else:
                with torch.no_grad():
                    f=self.former_net(t,n,a)
            u = f.detach()
            if self.memory_all:
                for j in range(0, y.shape[0]):
                    self.node_list[int(y[j])].nod_memory_generation_2(u[j].unsqueeze(0),t[j],n[j],t_f[j])

            else:
                for j in range(0, y.shape[0]):

                    self.node_list[int(y[j])].nod_memory_generation(u[j].unsqueeze(0),t[j],n[j],t_f[j])

        if self.memory_all:
            if self.random_mem:
                for w in range(0, len(self.leaf_dic)):
                    self.node_list[w].memory_wash_3()

            else:
                for w in range(0, len(self.leaf_dic)):
                    self.node_list[w].memory_wash_4()


    def tree_construction(self):

        for u in self.leaf_dic:

            node = memory_leaf_node(filter_value=self.filter_value, mean=self.mean,
                                    keep_input=self.keep_input,
                                    net=self.former_net,
                                    device=self.device, consistency=self.consistency,
                                    net_2=self.node_net, same_tansfer=self.same_transfer
                                    , random_number=self.random_number, distance=self.distance,mem_all=self.memory_all)
            node.meaning = u
            node.number = len(self.node_list)
            self.node_list.append(node)

        self.leaf_node_memory()
        for i in range(0, len(self.leaf_dic)):

            if self.memory_all:

                self.node_list[i].memory_f=[self.node_list[i].memory[mem][0]  for mem in range(len(self.node_list[i].memory))]
            else:
                self.node_list[i].memory_f=self.node_list[i].memory['extract_feature']

            self.node_list[i].memory_f=torch.cat(self.node_list[i].memory_f,dim=0)
        for j in self.condition:

            inner = inner_node(hidden_dim=13, out_dim=13, mean=self.mean
                               , keep_input=self.keep_input, net=self.former_net,
                               device=self.device, consistency=self.consistency,
                               net_2=self.node_net, same_transfer=self.same_transfer)
            inner.number = len(self.node_list)
            for k in j[0:len(j) - 1]:

                inner.children.append(self.node_list[k])
                self.node_list[k].father = inner

            inner.meaning = j[-1]
            self.node_list.append(inner)

        print("construction finished")
        return
    def show_memory_number(self):
        memory_number={}
        for i in range(len(self.leaf_dic)):
            memory_number[f'{i}']=self.node_list[i].memory_f.shape[0]
        print(memory_number)


    def tree_simulation_score_probility(self, input):

        if self.logic:
            for i in range(0, len(self.leaf_dic)):
                self.node_list[i].score = self.node_list[i].simulation_calculating(input, model).unsqueeze(0)
                # self.node_list[i].score = score.unsqueeze(0)
            for w in range(0, len(self.condition)):

                score_sum = 0
                for j in self.condition[w][0:-1]:
                    score_sum += self.node_list[j].score
                self.node_list[w + len(self.leaf_dic)].score = score_sum / (len(self.condition[w]) - 1)
                del score_sum
        else:


            for j in self.node_list[0:len(self.node_list) - 1]:
                j.score = j.simulation_calculating(input)

        for w in self.condition:
            """example of self.condition 
            [[0,3,6,9,'left'],[1,5,4,10 ,'right'],
        [2,7,8,'stright'],[13,14,15,'normal'],[11,12,'abnormal'],[16,17,'start']]
        'left' is the parent node for nodes 0, 3, 6, and 9.
        'right' is the parent node for nodes 1, 5, 4, and 10.
        Node numbers for 'left' are obtained by adding 13 to the corresponding numbers 
        (e.g., left node numbers are 13 + 0).
        Node numbers for 'right' are obtained by adding 13 to the corresponding numbers 
        (e.g., right node numbers are 13 + 1)."""
            concat_tensor = self.node_list[w[0]].score
            for k in w[1:len(w) - 1]:


                concat_tensor = torch.cat((concat_tensor, self.node_list[k].score))

            probility_list = F.softmax(concat_tensor, dim=0)
            for i in range(0, len(w) - 1):
                self.node_list[w[i]].node_probility = probility_list[i].unsqueeze(0)
            ###执行
            del probility_list
            del concat_tensor

    def tree_decision_probility(self, value):



        self.tree_simulation_score_probility(value)
        r = len(self.node_list) - 1
        while r > 12:

            for k in self.node_list[r].children:
                k.path_probility = self.node_list[r].path_probility * k.node_probility
            r = r - 1

        p1 = self.node_list[0].path_probility
        for t in self.node_list[1:13]:
            p1 = torch.cat((p1, t.path_probility), dim=0)


        return p1.t()




    def __call__(self, input, mean=False,node_noly=False):

        c = self.tree_decision_probility(input)
        return c

    def return_memory(self):

        for i in self.node_list[0:13]:
            if self.memory_all:

                yield i.memory[i.index.item()][1], i.memory[i.index.item()][2],i.memory[i.index.item()][3]
            else:

                 yield i.memory['raw_feature'][i.index.item()][0],i.memory['raw_feature'][i.index.item()][1],i.memory['raw_feature'][i.index.item()][2]



    def visualize(self, t, n, t_f, save_path="plot_save"):
        # Ensure the folder exists or create it
        import os
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        i = 0
        for target_trajectory, all_observations, t_f_1 in self.return_memory():
            if i >= 13:  # Stop after the first 14 plots
                break

            if not isinstance(target_trajectory, np.ndarray):
                target_trajectory = target_trajectory.cpu().numpy()

            fig, ax = plt.subplots(figsize=(5, 5))  # Create a new figure for each plot

            # Plot objective vehicle trajectory (purple color with hollow circles)
            target_x = target_trajectory[:, 2]
            target_y = target_trajectory[:, 3]
            ax.scatter(target_y, target_x, color='purple', s=2)
            ax.scatter(t_f_1[3], t_f_1[2], color='green',marker='x', s=2)
            # ax.scatter(t_f_1[3], t_f_1[2], edgecolorscolors='green', marker='x')


            # Separate data for different types and IDs
            for u in range(len(all_observations)):
                for j in range(all_observations[u].shape[0]):
                    x = all_observations[u][j, 2]
                    y = all_observations[u][j, 3]
                    obj_type = all_observations[u][j, 1]
                    obj_id = all_observations[u][j, 0]

                    if obj_type == target_trajectory[0, 1] and obj_id == target_trajectory[0, 0]:
                        continue

                    if obj_type == 1:
                        ax.scatter(y, x, facecolors='none', edgecolors='blue', s=2)
                    elif obj_type == 2:
                        ax.scatter(y, x, facecolors='none', edgecolors='red', s=2)
                    elif obj_type == 3:
                        ax.scatter(y, x, facecolors='none', edgecolors='black', s=2)

            ax.set_aspect('equal')
            ax.set_title(f'Object Trajectories for in memory of {self.node_list[i].meaning}', fontsize=5)
            ax.autoscale_view()
            ax.invert_xaxis()
            plt.tight_layout()
            # Save the plot to the specified directory with high quality
            plt.savefig(os.path.join(save_path, f"plot_{i}.png"), dpi=300)
            plt.close(fig)  # Close the current figure

            i += 1

        # Plotting for current scenario t, n, t_f
        if not isinstance(t, np.ndarray):
            t = t.cpu().numpy()

        fig, ax = plt.subplots(figsize=(5, 5))  # Create a new figure for current scenario

        # Plot objective vehicle trajectory (purple color with hollow circles)
        target_x = t[:, 2]
        target_y = t[:, 3]
        ax.scatter(target_y, target_x, facecolors='none', edgecolors='purple')
        # ax.scatter(t_f[1], t_f[0], facecolors='none', edgecolors='green', marker='x')

        # Plotting observations from n
        for frame in n:
            for j in range(frame.shape[0]):
                x = frame[j, 2]
                y = frame[j, 3]
                obj_type = frame[j, 1]
                obj_id = frame[j, 0]

                if obj_type == t[0, 1] and obj_id == t[0, 0]:
                    continue

                if obj_type == 1:
                    ax.scatter(y, x, facecolors='none', edgecolors='blue', s=2)
                elif obj_type == 2:
                    ax.scatter(y, x, facecolors='none', edgecolors='red', s=2)
                elif obj_type == 3:
                    ax.scatter(y, x, facecolors='none', edgecolors='black', s=2)

        ax.set_aspect('equal')
        ax.set_title('Object Trajectories for the current scenario', fontsize=5)
        plt.tight_layout()

        # Save the plot for current scenario to the specified directory with high quality
        plt.savefig(os.path.join(save_path, "current_scenario.png"), dpi=300)
        plt.close(fig)  # Close the current figure

        # After saving all the individual plots, create a new figure for the legend
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.axis('off')  # Turn off the axis

        legend_elements = [
            Line2D([0], [0], color='purple', marker='o', linestyle='None', label='Target Vehicle', markersize=4),
            Line2D([0], [0], color='red', marker='o', linestyle='None', label='Vehicle', markersize=5),
            Line2D([0], [0], color='black', marker='o', linestyle='None', label='Rider', markersize=5),
            Line2D([0], [0], color='blue', marker='o', linestyle='None', label='Pedestrian', markersize=5),
            Line2D([0], [0], color='green', marker='x', linestyle='None', label='Future Location', markersize=5)
        ]

        ax.legend(handles=legend_elements, loc='center')
        ax.autoscale_view()
        ax.invert_xaxis()
        plt.tight_layout()

        # Save the legend to the specified directory with high quality
        plt.savefig(os.path.join(save_path, "legend.png"), dpi=300)
        plt.close(fig)  # Close the legend figure


    def find_all_leaf_node(self,node):

        leaves = []
        if node.children is None:
            leaves.append(node)
            return leaves
        stack = [node]
        while stack:
            node = stack.pop()
            if node.children is not None:
                for child in node.children:
                    if child.children is None:  # if the child is a leaf node
                        leaves.append(child)
                    else:  # if the child is an inner node
                        stack.append(child)
        return leaves


    def show_decision_soft(self, input,t,n,t_f,how='visual'):
        if how not in ['visual']:
            raise RuntimeError('input erro')
        decision = []
        final = self.__call__(input, mean=False, model='train')
        score_list = [i.score[i.index] for i in self.node_list[:len(self.leaf_dic)]]
        decision.append([self.node_list[int(torch.argmax(final))].meaning,
            self.node_list[int(torch.argmax(final))].node_probility,int(torch.argmax(final))])
        pointer = int(torch.argmax(final))
        while self.node_list[pointer].father != None:
            pointer = self.node_list[pointer].father.number
            decision.append([self.node_list[pointer].meaning,self.node_list[pointer].node_probility,pointer])
        decision.reverse()
        print(decision)
        explain_dics={}
        if how=='visual':
            self.visualize(t,n,t_f)
            for i in decision[1:]:
                leaves=self.find_all_leaf_node(self.node_list[i[2]])
                explain_dics[i[0]] = [{leaf.meaning: leaf.score} for leaf in leaves]

            print(explain_dics)
            print('score:',score_list)
            return explain_dics
















