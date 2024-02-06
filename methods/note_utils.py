import random
import copy
import torch
import torch.nn.functional as F
import numpy as np


class PBRS():
    
    def __init__(self, capacity, num_class,random_seed=1,updateCriterion='Threshold'):
        #更新的criterion设置的类型为:
        #FIFO(先进先出),Threshold(阈值大小为优先级),random(随机撤走)

        random.seed(random_seed)

        self.data = [[[],[],[]] for _ in range(num_class)] #image, pseudo_cls
        #先获得num_class*[[],[]]

        self.counter = [0] * num_class #此变量记录迄今为止所有伪标签数量之和
        #[0,...,0]
        self.marker = [''] * num_class
        self.capacity = capacity
        self.num_class = num_class

        self.updateCriterion=updateCriterion

    def get_memory(self):
        data = self.data
        tmp_data = [[], []]
        for data_per_cls in data:
            feats, cls, _ = data_per_cls
            tmp_data[0].extend(feats)
            tmp_data[1].extend(cls)

        return tmp_data

    def get_occupancy(self):#获得self.data中所有数量之和
        occupancy = 0
        for data_per_cls in self.data:
            occupancy += len(data_per_cls[0])
        return occupancy

    def get_occupancy_per_class(self):#获得每个类别中存在内存中的数量之和
        occupancy_per_class = [0] * self.num_class
        for i, data_per_cls in enumerate(self.data):
            occupancy_per_class[i] = len(data_per_cls[0])
        return occupancy_per_class

    def add_instance(self, instance):#原始的数据可以用tensor,其他的两个可以单纯的使用数字
        #输入(instance=[原始的数据,类别标记,置信度])
        assert (len(instance) == 3)
        cls = instance[1]
        
        #print("cls:",cls)
        self.counter[cls] += 1 #进行计数 
        is_add = True

        if self.get_occupancy() >= self.capacity:
            is_add = self.remove_instance(cls,instance[2])

        if is_add:
            for i, dim in enumerate(self.data[cls]):
                dim.append(instance[i])

    def get_largest_indices(self):#返回所有并列最大值，组成一个列表

        occupancy_per_class = self.get_occupancy_per_class()
        max_value = max(occupancy_per_class)#max返回列表的最大值
        largest_indices = []
        for i, oc in enumerate(occupancy_per_class):
            if oc == max_value:
                largest_indices.append(i)
        return largest_indices

    def remove_instance(self, cls,confidence):#这个函数重写了，需要检查一下
        largest_indices = self.get_largest_indices()
        if cls not in largest_indices: #  instance is stored in the place of another instance that belongs to the largest class
            largest = random.choice(largest_indices)  # select only one largest class
            #print(f"origin:{self.data[largest][2]}")
            #random.choice返回列表中的一个随机项
            if self.updateCriterion=='random':
                tgt_idx = random.randrange(0, len(self.data[largest][0]))  # target index to remove
                #选择一个给定范围的随机项
            elif self.updateCriterion=='FIFO':
                tgt_idx=0
            elif self.updateCriterion=='Threshold':
                tgt_idx=self.data[largest][2].index(min(self.data[largest][2]))#选择置信度最小的
            else:
                raise ValueError("没有这种撤走样本的类型")
            for dim in self.data[largest]:
                dim.pop(tgt_idx)
            
            #print(f"modify:{self.data[largest][2]}")
        else:# replaces a randomly selected stored instance of the same class
            #print(f"origin:{self.data[cls][2]}")
            if self.updateCriterion=='random':
                m_c = self.get_occupancy_per_class()[cls]#cls为标签存在buffer的数量
                n_c = self.counter[cls]
                u = random.uniform(0, 1)
                if u <= m_c / n_c:
                    tgt_idx = random.randrange(0, len(self.data[cls][0]))  # target index to remove
                    for dim in self.data[cls]:
                        dim.pop(tgt_idx)
                else:
                    return False
            elif self.updateCriterion=='FIFO':
                for dim in self.data[cls]:
                    dim.pop(0)
            elif self.updateCriterion=='Threshold':
                minThreshold=min(self.data[cls][2])
                if confidence<minThreshold:
                    return False
                else:
                    tgt_idx=self.data[cls][2].index(minThreshold)
                    for dim in self.data[cls]:
                        dim.pop(tgt_idx)
            else:
                raise ValueError("没有这种撤走样本的类型")
            #print(f"modify:{self.data[cls][2]}")
        return True 
    

