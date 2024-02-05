
"""

blvd:1:IDs of objects;2:Driving scenario category of ego vehicle;3:Behaviour categories of ego vehicle ;
     4:Object categories: pedestrian(1), vehicle(2), and rider(3);5:Interactive behaviour;
     6-11:The location information of 3D bounding box, containing the 3D box and direction;
     12:The height value of the top surface of 3D bounding box;13:The height value of the bottom surface of 3D bounding box;

"""

import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random
import copy
import re
os.environ['PYTHONHASHSEED'] = str(42) #

class dataset():
    def __init__(self,future,input_size=5,trajectory=False,data_path_root=r'D:\pythonProject4\BLVD\data',
                ):

        self.future=future
        self.input_size=input_size
        self.data_path_root=data_path_root
        self.trajectory=trajectory
        self.dir= os.listdir(self.data_path_root)


        self.dirlist = [item for item in self.dir if os.path.isdir(os.path.join(self.data_path_root, item))]
    def data_process(self,dir):


        label_dir = os.path.join(self.data_path_root, dir, 'Label')
        file_list=os.listdir(label_dir)[0:-1]
        s = [j for j in os.listdir(os.path.join(self.data_path_root, dir)) if re.match(r'^\d+\.jpg$',j)]

        def extract_number(filename):

            match = re.match(r'\d+', filename)
            if match:
                return int(match.group())
            return 0
        s.sort(key=extract_number)
        image_path=[os.path.join(self.data_path_root,dir,i) for i in s]
        if len(file_list) == 0:
            print(dir)
            print('kong')
            return 0, [],[]
        file_list.sort(key=extract_number)
        if not file_list[-1].startswith('0'):

            file_list.pop()



        frame_list_size=int(file_list[-1].split('.')[0])

        frame_list = np.ones(frame_list_size).tolist()

        frame_list_2 =[]
        all_frame=0
        for filename in file_list:
            if filename.endswith('txt'):
                file_path = os.path.join(label_dir, filename)
                ###下面来加载文件
                frame_id = int(filename.split('.')[0])
                if os.path.getsize(file_path) != 0:


                    try:
                        data = np.loadtxt(file_path, dtype=np.float32, ndmin=2)  # ndmin  强制要求得到2维数组

                    except Exception as e:
                        print(file_path)
                        print(e.args)
                    if data.shape[1] == 13:

                        if len(data[(data[:, 3] == 3) & (data[:, 4] > 7), 4]) > 0:
                            data[(data[:, 3] == 3) & (data[:, 4] > 7), 4] = np.nan
                            data = pd.DataFrame(data)
                            data = data.dropna(axis=0, how='any')
                            data = data.to_numpy()
                        if len(data[(data[:, 3] == 1) & (data[:, 4] > 8), 4]) > 0:
                            data[(data[:, 3] == 1) & (data[:, 4] > 8), 4] = np.nan
                            data = pd.DataFrame(data)
                            data = data.dropna(axis=0, how='any')
                            data = data.to_numpy()
                        if len(data[(data[:, 3] == 2) & (data[:, 4] > 13), 4]) > 0:
                            data[(data[:, 3] == 2) & (data[:, 4] > 13), 4] = np.nan
                            data = pd.DataFrame(data)
                            data = data.dropna(axis=0, how='any')
                            data = data.to_numpy()
                        if data.size==0:

                            print('empty after clear')
                            print(filename)
                            continue
                        data = pd.DataFrame(data,
                                            columns=['IDs of objects', 'Driving scenario category of ego vehicle',
                                                     'Behaviour categories of ego vehicle', 'Object categories',
                                                     'Interactive behaviour', '3D_bounding_box_1',
                                                     '3D_bounding_box_2',
                                                     '3D_bounding_box_3', '3D_bounding_box_4', '3D_bounding_box_5',
                                                     '3D_bounding_box_6', '3D_bounding_box_7', '3D_bounding_box_8'])

                        data['frame_id'] = frame_id
                        data.dropna(axis=0, how='any')
                        if self.trajectory:
                            data = self.calculate_new_columns(data)

                            """

                            In order to distinguish better, here, the ID of the ego vehicle is set to -1.
                            """
                            frame_list[int(filename.split('.')[0])-1]=np.vstack((np.array([-1,2,0,0,0,0]),data.iloc[:,[0,3,5,6,7,8]].to_numpy()))
                            frame_list_2.append(data)
                        else:
                            frame_list[int(filename.split('.')[0])-1] =np.vstack((np.array([-1,2,0,0,0,0,0,0]), data.iloc[:, [0,3, 5, 6, 7, 8,9,10]].to_numpy()))
                            frame_list_2.append(data)
                    else:

                        print(filename)
                        print('The file is not thirteen cases')
                else:
                    ##谁是空文件
                    print(filename)
                    print('The file is empty')
        if len(frame_list_2)!=0:
            all_frame = pd.concat(frame_list_2)
            all_frame=all_frame.reset_index(drop=True)

        if isinstance(all_frame,pd.DataFrame):

            series = all_frame['frame_id'].copy()
            # 对 Series 进行去重操作
            unique_values = series.unique()
            # 将去重后的结果转换为列表
            unique_list = unique_values.tolist()
            for index in unique_list:

                if not isinstance(frame_list[index-1],np.ndarray):
                   raise  RuntimeError('数据加载出错')
        return all_frame,frame_list,image_path

    def frame_data_get(self,all_frame:pd.DataFrame,frame_list:list,data_ss:list,image_path:list):


        #首先对all_frame进行处理
        if isinstance(all_frame,int):
            return data_ss
        if len(frame_list) == 0:
            print(frame_list)
            return data_ss
        max_index=all_frame['frame_id'].max()
        for i in range(1,max_index+1):

            if (i+self.future+self.input_size-1)<=max_index:

                maby_target=all_frame.loc[(i<=all_frame['frame_id'])&(all_frame['frame_id']<=(i-1)+self.input_size+self.future),:]

                for _,k in maby_target.groupby(['IDs of objects','Object categories']):

                    if (k.shape[0]==(self.input_size+self.future))and(all(x==2 for x in k['Object categories'].tolist())):
                        #找到了合适的目标进入轨迹切割，node构建，ad_matrix构建阶段
                        if self.trajectory:

                            trajectory=k.iloc[:self.input_size,[0,3,5,6,7,8]].to_numpy()
                            trajectory_final=k.iloc[self.input_size:self.input_size+self.future,[5,6,7]].to_numpy()
                        else:
                            trajectory = k.iloc[:self.input_size, [0,3,5,6,7,8,9,10]].to_numpy()
                            trajectory_final=k.iloc[self.input_size:self.input_size+self.future, [5,6,7,8,9,10]].to_numpy()
                        label=k.iloc[self.input_size+self.future-1,4]-1
                        if (i+self.input_size)-1-(i-1)!=5:
                           raise RuntimeError('Sliding window configuration error')

                        if self.trajectory:
                            node=frame_list[int(k.iloc[0,9])-1:int(k.iloc[0,9])+self.input_size-1]
                            image = image_path[int(k.iloc[0, 9]) - 1:int(k.iloc[0, 9]) + self.input_size - 1]
                        else:
                            node=frame_list[int(k.iloc[0,13])-1:int(k.iloc[0,13])+self.input_size-1]

                            image=image_path[int(k.iloc[0,13])-1:int(k.iloc[0,13])+self.input_size-1]

                        if (len(node)!=self.input_size) or (not all(isinstance(x, np.ndarray) for x in node)):
                            raise RuntimeError('There is an issue with data extraction')

                        ad_matrix=[np.ones((w.shape[0],w.shape[0])) for w in node]
                        data_ss.append([trajectory,node,ad_matrix,image,label,trajectory_final])
            else:
                break
        return data_ss






    def calculate_new_columns(self,dataframe):

        t = [
            dataframe['3D_bounding_box_1'],
            dataframe['3D_bounding_box_2'],
            (dataframe['3D_bounding_box_7'] + dataframe['3D_bounding_box_8']) / 2
        ]

        sin_ry = dataframe['3D_bounding_box_5'] / np.sqrt(dataframe['3D_bounding_box_5'] *dataframe['3D_bounding_box_5'] + dataframe['3D_bounding_box_6'] * dataframe['3D_bounding_box_6'])
        cos_ry = dataframe['3D_bounding_box_6'] / np.sqrt(dataframe['3D_bounding_box_5'] * dataframe['3D_bounding_box_5'] +dataframe['3D_bounding_box_6'] * dataframe['3D_bounding_box_6'])

        ry = np.where(sin_ry >= 0, np.arccos(cos_ry), (-1) * np.arccos(cos_ry))

        s=pd.DataFrame({
            'IDs of objects':dataframe['IDs of objects'],
            'Driving scenario category of ego vehicle':dataframe['Driving scenario category of ego vehicle'],
            'Behaviour categories of ego vehicle': dataframe['Behaviour categories of ego vehicle'],
            'Object categories':dataframe['Object categories'],
            'Interactive behaviour':dataframe['Interactive behaviour'],
            't0': t[0],
            't1': t[1],
            't2': t[2],
            'ry': np.rad2deg(ry),
            'frame_id':dataframe['frame_id']
        })
        s.dropna(axis=0,how='any',inplace=True)#防止出现nan导致后续无法正常运行
        return s
    def __call__(self):

        data_ss_1=[]
        for j in self.dirlist:
            all_frame_1,frame_list_1,image=self.data_process(j)
            data_ss_1 = self.frame_data_get(all_frame=all_frame_1, frame_list=frame_list_1, data_ss=data_ss_1
                                            ,image_path=image)


        train_data, test_data = train_test_split(data_ss_1, test_size=0.2 , random_state=42)
        return train_data, test_data







class data_loder():
    def __init__(self,batch_size,drop=False):
        self.batch=batch_size
        self.drop=drop

    def __call__(self,data_set):

        length=len(data_set)
        if length==0:
            raise  RuntimeError('Please provide the correct data_set, and refer to the detailed'
                                ' data structure in the function description.')
        all_data=[]
        index=0
        count=0
        trajectory=[]
        node=[]
        ad_matrix=[]
        label=[]
        for t,n,a,l in data_set:

            trajectory.append(t)
            node.append(n)
            ad_matrix.append(a)
            label.append(l)
            count+=1
            index+=1
            if index<length and count==self.batch:

                if len(n) == 0 or len(a) == 0:

                    raise RuntimeError('There is an issue with data extraction.')
                all_data.append([torch.tensor(np.array(trajectory)),node.copy(),ad_matrix.copy(),torch.tensor(label)])

                trajectory.clear()
                node.clear()
                ad_matrix.clear()
                label.clear()
                count=0

            if (index==length and count<self.batch) and ( not self.drop) :

                all_data.append([torch.tensor(np.array(trajectory,dtype=np.float32)), node, ad_matrix, torch.tensor(label)])###加载剩余的数据
                break
            if (index == length and count < self.batch) and  self.drop :

                break
        return all_data


class DataLoader:
    def __init__(self, batch_size, drop=False):
        self.batch_size = batch_size
        self.drop = drop

    def __call__(self, data_set):

        trajectory = []
        node = []
        ad_matrix = []
        image_s=[]
        label = []
        trajectory_final=[]
        for t, n, a, image,l ,t_f in data_set:
            # 将数据添加到临时列表中
            trajectory.append(t)
            node.append(n)
            ad_matrix.append(a)
            image_s.append(image)
            label.append(l)
            trajectory_final.append(t_f)


            if len(node) == self.batch_size:

                yield torch.tensor(trajectory), node.copy(), ad_matrix.copy(), image_s.copy(),torch.tensor(label),torch.tensor((trajectory_final.copy()))
                # 清空临时列表
                trajectory.clear()
                node.clear()
                ad_matrix.clear()
                image_s.clear()
                label.clear()
                trajectory_final.clear()


        if not self.drop and len(node) > 0:
            yield torch.tensor(np.array(trajectory,dtype=np.float32)), node, ad_matrix, image_s.copy(),torch.tensor(label),\
                  torch.tensor(trajectory_final.copy())



class data_strong():

    def __init__(self,random_rate,data):
        self.random_rate=random_rate
        self.data=data
    def oversample(self):



        random.seed(self.random_rate)

        data_dict={}
        _,self.label=self.data_statistic()
        for j in self.label:

            data_dict['%d' % int(j)]=[]
        for t,n,a,image,l ,t_f in self.data:
            ####加载对应的label 的数据的list
            data_dict['%d' % int(l)].append([t,n,a,image,l,t_f])
        #初始化长度list
        statistic_list=np.array(self.label)
        for k,value in data_dict.items():

            statistic_list[int(k)]=len(data_dict[k])


        max_length=np.max(statistic_list)
        sample_=max_length-statistic_list
        oversampled_data = copy.deepcopy(self.data)
        for q in range(0,len(self.label)):
            sampel_len=sample_[q]
            sample_list=copy.deepcopy(random.choices(data_dict['%d' % q],k=sampel_len))
            oversampled_data += sample_list
        self.data=copy.deepcopy(oversampled_data)
        random.shuffle(self.data)
        return self.data

    def data_statistic(self):

        data_dict = {}
        label_list=[]
        for t, n, a, image,l,t_f in self.data:
            if l in label_list:

                data_dict['%d' % int(l)]+=1
            else:

                label_list.append(int(l))
                data_dict['%d' % int(l)]=1
        label_list.sort()
        sorted_dict = dict(sorted(data_dict.items(), key=lambda x: x[0]))
        print(sorted_dict)
        print(label_list)
        return sorted_dict,label_list



