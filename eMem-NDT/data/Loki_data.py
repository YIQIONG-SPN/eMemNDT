import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random
import copy
import re
from sklearn.preprocessing import LabelEncoder
import math
os.environ['PYTHONASHSEED']=str(42)
class data_loki():
    def __init__(self,future,input_size,data_path_root,label_need,label_find=False,intention=False):
        ##直接继承之前的数据集
        self.future=future
        self.input_size=input_size
        self.data_path_root=data_path_root
        self.dir= os.listdir(self.data_path_root)

        self.dirlist = [item for item in self.dir if os.path.isdir(os.path.join(self.data_path_root, item))]
        self.label_find=[]
        self.find=label_find
        self.intention=intention
        self.v_intention_label= ['Stopped' ,'Parked', 'Turn left', 'Turn right',
                                                  'Drive forward', 'Cut in to the left',
                                                  'Cut in to the right', 'Lane change to the left',
                                                  'Lane change to the right']
        self.mapping={'Stopped': 0, 'Parked': 1, 'Turn left': 2, 'Turn right': 3,
         'Cut in to the left': 4,
         'Cut in to the right': 5, 'Lane change to the left': 6,
         'Lane change to the right': 7,'Drive forward':8 }
        self.label_need=label_need

    def extract_label3d_strings(self,string_lists):
        label3d_pattern = r'label3d_\d+\.txt'
        extracted_strings = []

        for string_list in string_lists:

            match = re.search(label3d_pattern, string_list)
            if match:
                extracted_strings.append(match.group())

        return extracted_strings

    def extract_label3d_number(self,filename):
        label3d_pattern = r'label3d_(\d+)\.txt'
        match = re.search(label3d_pattern, filename)
        if match:
            return int(match.group(1))
        else:
            return None
    def data_process(self,dir):
        path=os.path.join(self.data_path_root,dir)
        dir_list=os.listdir(path)
        data_list=self.extract_label3d_strings(dir_list)
        if len(data_list) == 0:
            print(dir)
            print('kong')
            return 0, []

        data_list.sort() # 进行一次排序方便后续的处理
        frame_list_size=int(self.extract_label3d_number(data_list[-1])/2)#
        frame_list = np.ones(frame_list_size+1).tolist()
        frame_list_2 = []
        all_frame = 0
        for filename in data_list:
            if filename.endswith('txt'):
                file_path = os.path.join(path, filename)
                ###下面来加载文件
                frame_id = int(self.extract_label3d_number(filename)/2 )
                if os.path.getsize(file_path) != 0:


                    try:
                        df = pd.read_csv(file_path, sep=',',
                                         header=0)
                        if df.shape[1] !=14:
                            print(filename,'不是14列')
                            continue
                        labels = ['Pedestrian', 'Car', 'Bus', 'Truck', 'Van', 'Motorcyclist', 'Bicyclist']
                        df = df[df["labels"].isin(labels)]
                        mapping = {'Pedestrian': 0, 'Car': 1, 'Bus': 1, 'Truck': 1, 'Van': 1,
                                   'Motorcyclist':2,
                                   'Bicyclist': 3}

                        mapping2 = {'Stopped':0, 'Parked':1, 'Turn left':2, 'Turn right':3,
                                                  'Drive forward':4, 'Cut in to the left':5,
                                                  'Cut in to the right':6, 'Lane change to the left':7,
                                                  'Lane change to the right':8}
                        df["labels"] = df["labels"].map(mapping)
                        df.dropna(axis=0, how='any')
                        if not self.find:
                            df[' vehicle_state']=df[' vehicle_state'].map(mapping2)

                        if df.empty:
                            print(filename,'Empty after cleaning')
                            continue
                    except Exception as e:
                        print(file_path)
                        print(e.args)

                    df['frame_id'] = frame_id

                    data=df.iloc[:, [1, 0, 3, 4, 5, 9, 10,14,11]]



                    frame_list[int(frame_id)]=data.iloc[:,[1,0,2,3,4,5]]
                    frame_list_2.append(data)

                else:
                    ##谁是空文件
                    print(filename)
                    print('文件是空的')
        if len(frame_list_2)!=0:
            all_frame = pd.concat(frame_list_2)
            all_frame=all_frame.reset_index(drop=True)

        if isinstance(all_frame,pd.DataFrame):
            ##对frame_list进行检测
            series = all_frame['frame_id'].copy()
            # 对 Series 进行去重操作
            unique_values = series.unique()
            # 将去重后的结果转换为列表
            unique_list = unique_values.tolist()
            for index in unique_list:

                if not isinstance(frame_list[index],pd.DataFrame):
                   raise  RuntimeError('数据加载出错')
        return all_frame,frame_list






    def frame_data_get(self,all_frame:pd.DataFrame,frame_list:list,data_ss:list):

        #首先对all_frame进行处理
        if isinstance(all_frame,int):
            return data_ss
        if len(frame_list) == 0:
            print(frame_list)
            return data_ss
        max_index=all_frame['frame_id'].max()
        for i in range(0,max_index+1):
            ###开始依照i进行索引
            if (i+self.future+self.input_size)<=max_index:

                maby_target=all_frame.loc[(i<=all_frame['frame_id'])&(all_frame['frame_id']<=(i)+self.input_size+self.future),:]

                for _,k in maby_target.groupby(['labels',' track_id']):

                    if (k.shape[0]==(self.input_size+self.future))and(all(x==1 for x in k['labels'].tolist())):

                        trajectory=k.iloc[:self.input_size,[0,1,2,3,4,5]]
                        trajectory_final=k.iloc[self.input_size+self.future-1,[2,3,4,5]].to_numpy()
                        if not self.intention:
                            label=k.iloc[self.input_size+self.future-1,6]
                            if label not in self.v_intention_label:
                                continue
                            else:
                                label_1=float(self.mapping[label])
                                if label_1 not in self.label_need:
                                    continue
                        else:
                            label=k.iloc[self.input_size+self.future-1,8]
                        if not self.find:
                            if not isinstance(label,float)  :
                                continue
                            if math.isnan(label):
                                print('vehicle_state nan')
                                continue
                        else:
                            if label not in self.label_find:
                                self.label_find.append(label)
                        if (i+self.input_size)-(i)!=self.input_size:
                           raise RuntimeError('Sliding window configuration error')


                        node=frame_list[int(k.iloc[0,7]):int(k.iloc[0,7])+self.input_size]

                        if (len(node)!=self.input_size) or (not all(isinstance(x, pd.DataFrame) for x in node)):
                            raise RuntimeError('There is an issue with data extraction')

                        ad_matrix=[np.ones((w.shape[0]+1,w.shape[0]+1)) for w in node]
                        trajectory_processed,node_processed=self.track_id_mapping(trajectory,node)

                        data_ss.append([trajectory_processed.to_numpy(),node_processed,ad_matrix,label_1,trajectory_final])
            else:
                break
        return data_ss

    def track_id_mapping(self,trajectory,node):




        trajectory['frame_idx'] = 0
        for i, df in enumerate(node):
            df['frame_idx'] = i + 1


        all_frames = [trajectory] + node
        combined_df = pd.concat(all_frames, ignore_index=True)


        le = LabelEncoder()
        combined_df[combined_df.columns[0]]= le.fit_transform(combined_df[combined_df.columns[0]])#track_id


        trajectory_processed = combined_df[combined_df['frame_idx'] == 0].drop(columns=['frame_idx'])
        node_processed = [np.vstack((np.array([-1,1,0,0,0,0]),combined_df[combined_df['frame_idx'] == i].drop(columns=['frame_idx']))) for i in
                          range(1, len(node) + 1)]

        return trajectory_processed, node_processed
    def __call__(self):
        """
        :return:
        """
        data_ss_1=[]
        self.dirlist.sort()  ##  linux special
        for j in self.dirlist:
            all_frame_1,frame_list_1=self.data_process(j)
            data_ss_1 = self.frame_data_get(all_frame=all_frame_1, frame_list=frame_list_1, data_ss=data_ss_1)
        if self.find:
            print(self.label_find)

        train_data, test_data = train_test_split(data_ss_1, test_size=0.2 , random_state=42)
        return train_data,test_data
class DataLoader:
    def __init__(self, batch_size, drop=False):
        self.batch_size = batch_size
        self.drop = drop

    def __call__(self, data_set):

         trajectory = []
         node = []
         ad_matrix = []

         label = []
         trajectory_final = []
         for t, n, a,  l, t_f in data_set:

            trajectory.append(t)
            node.append(n)
            ad_matrix.append(a)
            label.append(l)
            trajectory_final.append(t_f)

            if len(node) == self.batch_size:

                yield torch.tensor(
                    np.array(trajectory)), node.copy(), ad_matrix.copy(), torch.tensor(
                    label), trajectory_final.copy()


                trajectory.clear()
                node.clear()
                ad_matrix.clear()

                label.clear()
                trajectory_final.clear()


         if not self.drop and len(node) > 0:
            yield torch.tensor(np.array(trajectory)), node, ad_matrix,  torch.tensor(
                label), trajectory_final.copy()

class data_strong():

    def __init__(self,random_rate,data):
        self.random_rate=random_rate
        self.data=data
    def oversample(self):

        random.seed(self.random_rate)

        data_dict={}
        _,self.label=self.data_statistic()
        self.label = [i for i in range(9)]
        for j in self.label:
            ####初始化字典
            data_dict['%d' % int(j)]=[]

        for t,n,a,l ,t_f in self.data:
            ####加载对应的label 的数据的list
            data_dict['%d' % int(l)].append([t,n,a,l,t_f])
        #初始化长度list
        statistic_list=np.array(self.label)
        for k,value in data_dict.items():


            statistic_list[int(k)]=len(data_dict[k])


        max_length=np.max(statistic_list)
        sample_=max_length-statistic_list
        oversampled_data = copy.deepcopy(self.data)


        for q in range(0,len(self.label)):
            sampel_len=sample_[q]
            if sampel_len==max_length:
                continue
            sample_list=copy.deepcopy(random.choices(data_dict['%d' % q],k=sampel_len))
            oversampled_data += sample_list
        self.data=copy.deepcopy(oversampled_data)
        random.shuffle(self.data)
        return self.data

    def data_statistic(self):

        data_dict = {}
        label_list=[]
        for t, n, a, l,t_f in self.data:
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