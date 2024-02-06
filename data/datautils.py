import os
from typing import Tuple
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from data.hoi_dataset import BongardDataset
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from data.fewshot_datasets import *
import data.augmix_ops as augmentations

from data.corruption_datasets import corruptions_datasets
from .cifar import *
from .tinyimagenet import *

ID_to_DIRNAME={
    'I': 'ImageNet',
    'A': 'imagenet-a',
    'K': 'ImageNet-Sketch',
    'R': 'imagenet-r',
    'V': 'imagenetv2-matched-frequency-format-val',
    'flower102': 'Flower102',
    'dtd': 'DTD',
    'pets': 'OxfordPets',
    'cars': 'StanfordCars',
    'ucf101': 'UCF101',
    'caltech101': 'caltech-101',
    'food101': 'Food101',
    'sun397': 'SUN397',
    'aircraft': 'fgvc-aircraft-2013b',
    'eurosat': 'eurosat'
}



def build_dataset(set_id, transform, data_root,args, mode='test', n_shot=None, split="all", bongard_anno=False,Image=True):
    #输入(数据集,transform,目录,相关模式(默认"test"),Image表示返回的testset的数据的类型是否为Image类型)
    #输出:测试集
    print("set_id:",set_id)
    if set_id == 'I':
        # ImageNet validation set
        testdir = os.path.join(os.path.join(data_root, ID_to_DIRNAME[set_id]), 'val')
        testset = datasets.ImageFolder(testdir, transform=transform)
    elif set_id in ['A', 'K', 'R', 'V']:
        testdir = os.path.join(data_root, ID_to_DIRNAME[set_id])
        print(testdir)
        tmptestset = datasets.ImageFolder(testdir, transform=transform)
        testset=[]
        testset.append(tmptestset)
        #print(f"\n{testdir}\n")
    elif set_id in fewshot_datasets:
        if mode == 'train' and n_shot:
            testset = build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform, mode=mode, n_shot=n_shot)
        else:
            testset = build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform, mode=mode)
    elif set_id == 'bongard':
        assert isinstance(transform, Tuple)
        base_transform, query_transform = transform
        testset = BongardDataset(data_root, split, mode, base_transform, query_transform, bongard_anno)
    
    elif set_id in corruptions_datasets:
        if set_id!='TinyImageNet_C':
            testset=build_CIFARC_dataset(set_id,transform,args,Image)#如果想跑CIFAR10-C的所有数据集的话,可以尝试生成一个testset列表
        else:
            testset=build_TinyImagenetC_dataset(set_id,transform,args)
        #testset = CIFAR10(root=os.path.expanduser("/data/zhangdingchu/data/data"), download=True, train=False)
        #testset=build_CIFAR_dataset(set_id,transform)
    else:
        raise NotImplementedError
        
    return testset


# AugMix Transforms
def get_preaugment():
    return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ])

def augmix(image, preprocess, aug_list, severity=1):
    preaugment = get_preaugment()
    x_orig = preaugment(image)
    x_processed = preprocess(x_orig)
    if len(aug_list) == 0:
        return x_processed
    w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
    m = np.float32(np.random.beta(1.0, 1.0))

    mix = torch.zeros_like(x_processed)
    for i in range(3):
        x_aug = x_orig.copy()
        for _ in range(np.random.randint(1, 4)):
            x_aug = np.random.choice(aug_list)(x_aug, severity)
        mix += w[i] * preprocess(x_aug)
    mix = m * x_processed + (1 - m) * mix
    return mix


class AugMixAugmenter(object):
    def __init__(self, base_transform, preprocess, n_views=2, augmix=False, 
                    severity=1):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views
        if augmix:
            self.aug_list = augmentations.augmentations
        else:
            self.aug_list = []
        self.severity = severity
        
    def __call__(self, x):
        image = self.preprocess(self.base_transform(x))
        views = [augmix(x, self.preprocess, self.aug_list, self.severity) for _ in range(self.n_views)]
        return [image] + views
    
class BaseTransform(object):
    def __init__(self, base_transform, preprocess):
        self.base_transform = base_transform
        self.preprocess = preprocess
        
    def __call__(self, x):
        image = self.preprocess(self.base_transform(x))
        return image



