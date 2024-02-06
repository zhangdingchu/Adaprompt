from ast import Num
import numpy as np
import logging, os
import torch
import math
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from .base import StreamDataset
from sklearn.utils import shuffle
import utils
from torch.utils.data import Dataset
from PIL import Image
from torchvision.datasets import CIFAR10


__all__ = ['CIFAR10C', 'CIFAR100C', "get_cifar_loader", "CIFAR10CB","build_CIFARC_dataset","build_CIFAR_dataset"]

def build_CIFAR_dataset(datasetName,transform):
    dataset=CIFAR10(root=os.path.expanduser("/data/zhangdingchu/data/data"), download=True, train=False)
    return CIFARDataset(dataset,transform)

class CIFARDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.image=[]
        self.targets = []
        self.transform = transform
        for data in dataset:
            self.image.append(data[0])
            self.targets.append(data[1])
        
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, index):
        image = self.image[index]
        target = self.targets[index]
        if self.transform is not None:
            image = self.transform(image)
        return image,target
    
    def tensor2PIL(self,ImageTensor):
        array = ImageTensor.permute(1, 2, 0).numpy()
        #array = ImageTensor.numpy()
        #print("array.shape:",array.shape)
        # 确保数组的数据类型在0到255之间，并转换为无符号整数类型
        array = np.uint8(np.clip(array, 0, 1) * 255)
        # 创建一个PIL.Image对象
        image = Image.fromarray(array)
        return image

def build_CIFARC_dataset(datasetName,transform,args,Image):
    if datasetName=='CIFAR10_C':
        dataset=CIFAR10C(root=args.stream.dataset_dir,
                seed=args.seed,
                batch_size=args.stream.batch_size,
                severities=args.stream.severities,#这里是一个列表[5]
                corruptions=args.stream.corruptions)
    elif datasetName=='CIFAR100_C':
        dataset=CIFAR100C(root=args.stream.dataset_dir,
                seed=args.seed,
                batch_size=args.stream.batch_size,
                severities=args.stream.severities,#这里是一个列表[5]
                corruptions=args.stream.corruptions)
    else:
        raise NotImplementedError
    
    imagesList,targetsList=dataset.load_corruption_dataset()

    DatasetC=[]

    for i in range(len(imagesList)):
        images=imagesList[i]
        targets=targetsList[i]
        DatasetC.append(CIFARCDataset(images,targets,transform,Image))

    return DatasetC

    #images,targets=dataset.load_corruption_dataset(0,0)
    
    #return CIFARCDataset(images,targets,transform,Image)

class CIFARCDataset(Dataset):
    def __init__(self, image, targets, transform=None,Image=True):
        self.image=image
        self.targets = targets
        self.transform = transform
        self.Image=Image
        
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, index):
        image = self.image[index]
        target = self.targets[index]
        #print("image:",type(image))
        if self.Image==True:
            image=self.tensor2PIL(image)#np.array(image)
        #print("image:",type(image))
        if self.transform is not None:
            image = self.transform(image)
        return image,target
    
    def tensor2PIL(self,ImageTensor):
        array = ImageTensor.permute(1, 2, 0).numpy()
        #array = ImageTensor.numpy()
        #print("array.shape:",array.shape)
        # 确保数组的数据类型在0到255之间，并转换为无符号整数类型
        array = np.uint8(np.clip(array, 0, 1) * 255)
        # 创建一个PIL.Image对象
        image = Image.fromarray(array)
        return image
    

class CIFAR10C(StreamDataset):
    NAME = "CIFAR-10-C" 
    CORRUPTIONS = ['gaussian_noise', 'shot_noise', 'impulse_noise', 
                   'defocus_blur', 'glass_blur', 'motion_blur',
                     'zoom_blur', 'snow', 'frost', 'fog', 
                     'brightness', 'contrast', 'elastic_transform', 
                     'pixelate', 'jpeg_compression']


    def __init__(self, 
        root="/data/zhangdingchu/data/data", 
        batch_size=64, 
        seed=998244353,
        corruptions=CORRUPTIONS,
        severities=[5],
    ):
        super().__init__(root, batch_size, seed)
        self._epoch, self._epochs = 0, 0
        self._data, self._label = None, None
        self.severities = severities if isinstance(severities, list) else [severities]
        self.corruptions = corruptions


    def run(self, ):
        for i_c, corruption in enumerate(self.corruptions):
            for i_s, severity in enumerate(self.severities):
                yield self._generate(i_c, i_s)
    
    def _generate(self, i_c, i_s):#yield产生出来的数值返回给其他变量使用
        corruption = self.corruptions[i_c]
        severity = self.severities[i_s]
        # Read & shuffle data
        data, label = self.load_corruptions_cifar(severity=severity, corruption=corruption)
        #print("shape:",data.shape,label.shape)
        rand_index = self.rand.permutation(data.shape[0])#随机返回一个[0,data.shape[0])的索引
        data, label = data[rand_index, ...], label[rand_index, ...]#相当于打乱数据
        print("shape:",data.shape,label.shape)
        # Calculate class ratio
        ratio = [torch.sum(label == i) / len(label) for i in range(self.num_classes())]#计算各类别比例
        logging.info(" > Class Ratio: ")
        logging.info(ratio)
        # Yield data
        n_batches = math.ceil(data.shape[0] / self.batch_size)#batch的数量
        for counter in range(n_batches):
            data_curr = data[counter * self.batch_size:(counter + 1) * self.batch_size]
            label_curr = label[counter * self.batch_size:(counter + 1) * self.batch_size]
            tag_cur = f"{corruption}-{severity}-{counter}"
            yield data_curr, label_curr, tag_cur, ratio#返回batch的数据和标签


    def num_classes(self,): 
        return 10

    def download(self,): pass

    def load_cifar(self,):
        dataset = datasets.CIFAR10(root=self.root, train=False, transform=transforms.Compose([transforms.ToTensor()]), download=True)
        return self.load_dataset(dataset)

    def load_corruptions_cifar(self, severity, corruption):#根据后两个参数返回相应的图像和labels(类型为tensor)
        assert 0 <= severity <= 5
        if severity > 0: assert corruption in self.CORRUPTIONS
        if severity == 0: return self.load_cifar()
        n_total_cifar = 10000
        if not os.path.exists(self.root): raise FileNotFoundError("The root of datasets is not found.")
        data_dir = os.path.join(self.root, self.NAME)

        #print("data_dir:",data_dir)
        # Load labels
        label_path = os.path.join(data_dir, "labels.npy")
        if not os.path.isfile(label_path): raise FileNotFoundError("Labels are missing.")
        labels = np.load(label_path)

        #print("labels:",labels.shape)
        labels = labels[: n_total_cifar]#cifar10-c总共有50000标签
        # Load images
        data_path = os.path.join(data_dir, f"{corruption}.npy")
        if not os.path.isfile(data_path): raise FileNotFoundError("Data {corruption} is missing.")
        images_all = np.load(data_path)
        images = images_all[(severity - 1) * n_total_cifar: severity * n_total_cifar]#不同severity存放在不同的位置上
        #print("images:",images.shape)
        # To torch tensors
        images = np.transpose(images, (0, 3, 1, 2))#transpose将第一个参数的shape按照第2个参数的元组进行替换，第二个参数为维度的顺序
        images = images.astype(np.float32) / 255#进行归一化操作
        images = torch.tensor(images)
        labels = torch.tensor(labels)
        #print("images:",images.shape)
        return images, labels
    
    def load_corruption_dataset(self):

        dataList,labelList=[],[]

        for i_c, corruption in enumerate(self.corruptions):
            for i_s, severity in enumerate(self.severities):
                data, label = self.load_corruptions_cifar(severity=severity, corruption=corruption)
                dataList.append(data)
                labelList.append(label)

        return dataList,labelList
    
    # def load_corruption_dataset(self, i_c, i_s):
    #     corruption = self.corruptions[i_c]
    #     severity = self.severities[i_s]
    #     # Read & shuffle data
    #     data, label = self.load_corruptions_cifar(severity=severity, corruption=corruption)
    #     return (data,label)

class CIFAR10CB(CIFAR10C):
    NAME = "CIFAR-10-C" 
    CORRUPTIONS = ("shot_noise", "motion_blur", "snow", "pixelate", 
        "gaussian_noise", "defocus_blur", "brightness", "fog",
        "zoom_blur", "frost", "glass_blur", "impulse_noise", "contrast",
        "jpeg_compression", "elastic_transform")

    def __init__(self, 
        root="/data", 
        batch_size=64, 
        seed=998244353,
        corruptions=CORRUPTIONS,
        severities=1,
        bind_class=[],
        bind_ratio=[],
    ):
        super().__init__(root=root, batch_size=batch_size, seed=seed, corruptions=corruptions, severities=severities)
        self.bind_class = bind_class
        self.bind_ratio = bind_ratio

    def _generate(self, i_c, i_s):
        corruption = self.corruptions[i_c]
        severity = self.severities[i_s]
        # Read
        data, label = self.load_corruptions_cifar(severity=severity, corruption=corruption)
        if i_c < len(self.bind_class):
            index = self.bind_class[i_c]
            ratio = self.bind_ratio
            n_samples = data.shape[0]
            proba = torch.where(label == index, ratio / n_samples * self.num_classes(), (1.0 - ratio) / n_samples / (self.num_classes() - 1))
            proba = proba / proba.sum()
            n_sampling = int(data.shape[0] / self.num_classes() / ratio)
            index = self.rand.choice(n_samples, n_sampling, p=proba.cpu().numpy(), replace=False)
            data, label = data[index, ...], label[index, ...]
        # Shuffle data
        rand_index = self.rand.permutation(data.shape[0])
        data, label = data[rand_index, ...], label[rand_index, ...]
        
        # Calculate class ratio
        ratio = [torch.sum(label == i) / len(label) for i in range(self.num_classes())]
        logging.info(" > Class Ratio: ")
        logging.info(ratio)
        
        # Yield data
        n_batches = math.ceil(data.shape[0] / self.batch_size)
        for counter in range(n_batches):
            data_curr = data[counter * self.batch_size:(counter + 1) * self.batch_size]
            label_curr = label[counter * self.batch_size:(counter + 1) * self.batch_size]
            tag_cur = f"{corruption}-{severity}-{counter}"
            yield data_curr, label_curr, tag_cur, ratio

class CIFAR100C(CIFAR10C):
    NAME = "CIFAR-100-C" 
    CORRUPTIONS = ['gaussian_noise', 'shot_noise', 'impulse_noise', 
                   'defocus_blur', 'glass_blur', 'motion_blur',
                     'zoom_blur', 'snow', 'frost', 'fog', 
                     'brightness', 'contrast', 'elastic_transform', 
                     'pixelate', 'jpeg_compression']

    def __init__(self, 
        root="/data/zhangdingchu/data/data", 
        batch_size=64, 
        seed=998244353,
        corruptions=CORRUPTIONS,
        severities=[5],
    ):
        super().__init__(root, batch_size, seed)
        self._epoch, self._epochs = 0, 0
        self._data, self._label = None, None
        self.severities = severities if isinstance(severities, list) else [severities]
        self.corruptions = corruptions

    def num_classes(self,): return 100

    def load_cifar(self,):
        dataset = datasets.CIFAR100(root=self.root, train=False, transform=transforms.Compose([transforms.ToTensor()]), download=True)
        return self.load_dataset(dataset)

def get_cifar_loader(dataset, root="/data", batch_size=256,transform_train=None,transform_test=None):#根据dataset来将数据load到root中,返回训练数据，测试数据，类别总数。
    assert dataset in ["CIFAR10", "CIFAR100"]
    NORM_VAL = {
        "CIFAR10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        "CIFAR100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    }
    if transform_train==None:
        print("yes")
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),#图片进行随机裁剪,后面那个参数为对边界的填充
            transforms.RandomHorizontalFlip(),#一半概率翻转一半不翻
            transforms.ToTensor(),#进行张量化操作
            #transforms.Normalize(NORM_VAL[args.dataset]),
        ])
    if transform_test==None:
        print("yes")
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(NORM_VAL[args.dataset]),
        ])
#hasattr(obj,name)->True or false判断对象obj是否有方法或属性name
#getattr(obj,name)->返回属性或方法的地址
#setattr(obj, name, value)->给类的属性添加值，无返回值
    trainset = getattr(datasets, dataset)(root=root, train=True, download=True, transform=transform_train)
    testset = getattr(datasets, dataset)(root=root, train=False, download=True, transform=transform_test)
#相当于 datasets.CIFAR10/100(root(数据集根目录),train=True(训练集或测试集),target_transform=None(接受图像转换的版本), download=False(是否下载到root))
    #print("trainset:",trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    #print("trainloader:",trainloader)
#dataset=加载的数据集，batch_size每个batch加载的样本数量，shuffle每个epoch会打乱数据,num_workers用多少个子进程加载数据

    CLAS_VAL = {
        "CIFAR10": 10,
        "CIFAR100": 100,
    }
    n_classes = CLAS_VAL[dataset]

    return trainloader, testloader, n_classes

