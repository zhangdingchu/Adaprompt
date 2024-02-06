import math
import os

import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import PIL
from PIL import Image


class BaseJsonDataset(Dataset):
    def __init__(self, image_path, json_path, mode='train', n_shot=None, transform=None):
        self.transform = transform
        self.image_path = image_path
        self.split_json = json_path
        self.mode = mode
        self.image_list = []
        self.label_list = []
        with open(self.split_json) as fp:
            splits = json.load(fp)
            samples = splits[self.mode]
            for s in samples:
                self.image_list.append(s[0])
                self.label_list.append(s[1])
    
        if n_shot is not None:
            few_shot_samples = []
            c_range = max(self.label_list) + 1
            for c in range(c_range):
                c_idx = [idx for idx, lable in enumerate(self.label_list) if lable == c]
                random.seed(0)
                few_shot_samples.extend(random.sample(c_idx, n_shot))
            self.image_list = [self.image_list[i] for i in few_shot_samples]
            self.label_list = [self.label_list[i] for i in few_shot_samples]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_path, self.image_list[idx])
        image = Image.open(image_path).convert('RGB')
        #print(type(image))
        label = self.label_list[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label).long()

fewshot_datasets = ['DTD', 'Flower102', 'food101', 
                    'Pets', 'Caltech101', 'UCF101', 'eurosat']

path_dict = {
    # dataset_name: ["image_dir", "json_split_file"]
    "flower102": ["102flowers/jpg", "/data/zhangdingchu/data/data/Flower102/split_zhou_OxfordFlowers.json"],
    "food101": ["images", "/data/zhangdingchu/data/data/Food101/split_zhou_Food101.json"],
    "dtd": ["images", "/data/zhangdingchu/data/data/DTD/split_zhou_DescribableTextures.json"],
    "pets": ["images/images", "/data/zhangdingchu/data/data/OxfordPets/split_zhou_OxfordPets.json"],
    "caltech101": ["101_ObjectCategories", "/data/zhangdingchu/data/data/caltech-101/split_zhou_Caltech101.json"],
    "ucf101": ["UCF-101-midframes", "/data/zhangdingchu/data/data/UCF101/split_zhou_UCF101.json"],
    "eurosat": ["2750", "/data/zhangdingchu/data/data/eurosat/split_zhou_EuroSAT.json"]
}

def build_fewshot_dataset(set_id, root, transform, mode='train', n_shot=None):
    if set_id.lower() == 'aircraft':
        return [Aircraft(root, mode, n_shot, transform)]
    path_suffix, json_path = path_dict[set_id.lower()]
    image_path = os.path.join(root, path_suffix)
    ans=[]
    ans.append(BaseJsonDataset(image_path, json_path, mode, n_shot, transform)) 
    return ans


class Aircraft(Dataset):
    """ FGVC Aircraft dataset """
    def __init__(self, root, mode='train', n_shot=None, transform=None):
        self.transform = transform
        self.path = root
        self.mode = mode

        print(f"root:{root}")
        print(f"path1:{os.path.join(self.path, 'data/variants.txt')}")
        print(f"path2:{os.path.join(self.path, 'data/images_variant_{:s}.txt'.format(self.mode))}")

        self.cname = []
        with open(os.path.join(self.path, "data/variants.txt"), 'r') as fp:
            self.cname = [l.replace("\n", "") for l in fp.readlines()]

        self.image_list = []
        self.label_list = []
        with open(os.path.join(self.path, 'data/images_variant_{:s}.txt'.format(self.mode)), 'r') as fp:
            lines = [s.replace("\n", "") for s in fp.readlines()]
            for l in lines:
                ls = l.split(" ")
                img = ls[0]
                label = " ".join(ls[1:])
                # print("{}.jpg".format(img))
                # print(self.cname.index(label))
                self.image_list.append("{}.jpg".format(img))
                self.label_list.append(self.cname.index(label))
        
        print(f"example:{self.image_list[0]} and {self.label_list[0]}")

        if n_shot is not None:
            few_shot_samples = []
            c_range = max(self.label_list) + 1
            for c in range(c_range):
                c_idx = [idx for idx, lable in enumerate(self.label_list) if lable == c]
                random.seed(0)
                few_shot_samples.extend(random.sample(c_idx, n_shot))
            self.image_list = [self.image_list[i] for i in few_shot_samples]
            self.label_list = [self.label_list[i] for i in few_shot_samples]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.path, 'data/images', self.image_list[idx])
        #print(image_path)
        image = Image.open(image_path).convert('RGB')
        label = self.label_list[idx]
        if self.transform:
            image = self.transform(image)
        if image.shape[0]==3 and image.shape[1]==224 and image.shape[2]==224:
            pass
        else:
            print(image.shape)
        return image, torch.tensor(label).long()

