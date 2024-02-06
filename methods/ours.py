from copy import deepcopy

import torch
import torch.nn as nn

import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms

from PIL import Image

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from data.datautils import AugMixAugmenter,BaseTransform, build_dataset
import time
from data.fewshot_datasets import fewshot_datasets

import numpy as np

from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight, set_random_seed
from data.cls_to_names import *

import torch.nn.functional as F

import matplotlib.pyplot as plt

from .note_utils import PBRS
from data.corruption_datasets import corruptions_datasets
from data.imagnet_prompts import imagenet_classes
from data.imagenet_ar import A,V, R,ImageNetdata

import utils

__all__ = ["setup"]

#目前性能不好可以调节的参数有:
#优化器的种类,优化器的学习率,选择样本的threshold

class ours(nn.Module):
    """Norm adapts a model by estimating feature statistics during testing.

    Once equipped with Norm, the model normalizes its features during testing with batch-wise statistics, just like batch norm does during training.
    """

    def __init__(self, model,args,logger):
        super().__init__()
        self.model = model
        self.args=args.method
        
        logger.info(f"=> Model created: visual backbone {args.arch}")
    
        if not torch.cuda.is_available():
            logger.info('using CPU, this will be slow')
        else:
            assert args.gpu is not None
            torch.cuda.set_device(args.gpu)
            self.model = self.model.cuda(args.gpu)
        cudnn.benchmark = True

        self.model_state = deepcopy(self.model.state_dict())
        #deepcopy()进行深拷贝，防止父对象和子对象公用同一个地址
        #self.mem=PBRS(self.args.queue_size,args.stream.num_class,updateCriterion=self.args.updateCriterion)

    def forward(self, x):
        return self.model(x)
    
    def test_time_tuning(self,x):
        pass

    def recordModelState(self):
        self.model_state = deepcopy(self.model.state_dict())

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)
    
    def get_parameters(self):
        #获得投影层和预测层的学习率参数
        # params = [
        #     {"params": self.basemodel.parameters(), "lr":  base_lr},
        #     {"params": self.pro_head.parameters(), "lr":  base_lr}
        # ]
        params=[]
        print(f"len:{len(self.model.promptLearnerList)}")
        for promptLearner in self.model.promptLearnerList:
            tmpDict={"params":promptLearner.parameters(),"lr":self.args.lr}
            params.append(tmpDict)
        return params



#目前正在写Fixed的形式
class FixtestData():
    def __init__(self, tptmodel,dataset,logger,args):
        self.tptmodel=tptmodel
        self.dataset=dataset
        self.logger=logger
        self.args=args
        self.initDataset(args,dataset,logger)
        self.initModel(args)
        self.update=True
        cudnn.benchmark = True

        #对于优化器,目前可以使用的优化器有SGD,Adam,AdamW
    
    def initModel(self,args):
        
        for name, param in self.tptmodel.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)
        
        self.originmodel=deepcopy(self.tptmodel)

        # self.tptmodel.recordModelState()

        self.mem=PBRS(args.method.queue_size,args.stream.num_class,updateCriterion=args.method.updateCriterion)

        self.chooseThreshold=args.method.chooseThreshold

        self.criterion=nn.CrossEntropyLoss()

        self.optimizer=torch.optim.AdamW(self.tptmodel.get_parameters(),args.method.lr)
        #self.optimizer=torch.optim.AdamW(self.tptmodel.model.promptLearnerList[0].parameters(),args.method.lr)
        #注意这里只优化了第一个prompt
        #这个优化器有问题
        self.optim_state = deepcopy(self.optimizer.state_dict())
        self.scaler = torch.cuda.amp.GradScaler(init_scale=1000)
        #对于多个网络的参数可以使用[{'params': params1},{'params': first_params}]
        #对于优化器,目前可以使用的优化器有SGD,Adam,AdamW

    def initDataset(self,args,dataset,logger):
        normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

        base_transform = transforms.Compose([
                transforms.Resize(args.resolution, interpolation=BICUBIC),
                transforms.CenterCrop(args.resolution)])
        preprocess = transforms.Compose([
                transforms.ToTensor(),
                normalize])
        data_transform = BaseTransform(base_transform, preprocess)

        batchsize = args.stream.batch_size

        print(f"dataset:{dataset}")

        if dataset in fewshot_datasets:
            classnames = eval("{}_classes".format(dataset.lower()))
        elif args.test_sets in corruptions_datasets:
            classnames=eval("{}_classes".format(dataset))
        elif dataset in ImageNetdata:
            classnames=eval("{}".format(dataset))
        else:
            classnames = imagenet_classes
        
        logger.info(f"classnames: {classnames}")
        self.tptmodel.model.reset_classnames(classnames, args.arch)

        val_dataset_list = build_dataset(dataset, data_transform, args.data,args, mode=args.dataset_mode,Image=True)

        logger.info(f"number of test domain: {len(val_dataset_list)}")

        self.val_loader_list=[]

        for val_dataset in val_dataset_list:
            val_loader = torch.utils.data.DataLoader(
                        val_dataset,
                        batch_size=batchsize, shuffle=True,
                        num_workers=args.workers, pin_memory=True)
            self.val_loader_list.append(val_loader)

    def recordResult(self):
        
        for idx, val_loader in enumerate(self.val_loader_list):

            batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
            top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
            top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

            progress = ProgressMeter(
                len(val_loader),
                [batch_time, top1, top5],
                self.logger,
                prefix='Test: ')

            # reset model and switch to evaluate mode
            self.tptmodel.model.eval()
            self.originmodel.model.eval()
            
            allOutput=[]
            allTarget=[]
            
            wall_time=0
            for i, (images, target) in enumerate(val_loader):
                
                images=images.cuda(self.args.gpu, non_blocking=True)
                target=target.cuda(self.args.gpu, non_blocking=True)
                # The actual inference goes here
                
                torch.cuda.synchronize()
                start_time = time.time()
                #将样本存入缓冲区里
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        output = self.originmodel(images)
                        output=output.mean(0)
                        output=F.softmax(output,dim=1)
                        
                maxProbas,maxIdx=output.max(1)

                allOutput.append(output)

                for j in range(maxProbas.shape[0]):
                    if maxProbas[j].item()>self.chooseThreshold:
                        self.mem.add_instance([images[j].unsqueeze(0),maxIdx[j].item(),maxProbas[j].item()])
                

                #取出相应的样本并进行更新
                tmp_data,tmp_label=self.mem.get_memory()#这里不用伪标注,用熵更好一点

                memorySize=len(tmp_data)
                #print(f"labelNum:{tmp_data[1]}")
                
                if memorySize>=self.args.method.queue_size and self.update==True:
                    #这里需要修改一下,当类别数量少的时候其实可以提前更新
                    tmp_data=torch.cat(tmp_data)
                    tmp_label=torch.tensor(tmp_label)

                    updateOutput = self.tptmodel(tmp_data)
                    if self.args.prompt==-1:
                        MeanOutput=updateOutput.mean(0)#这里也可以尝试尝试先softmax然后再mean
                    else:
                        MeanOutput=updateOutput[self.args.prompt]##↑是集成
                    MeanOutput=F.softmax(MeanOutput,dim=1)

                    loss=self.criterion(MeanOutput,tmp_label.cuda(self.args.gpu, non_blocking=True))
                    
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        output = self.tptmodel(images)
                        if self.args.prompt==-1:
                            output=output.mean(0)
                        else:
                            output=output[self.args.prompt]##↑是集成
                torch.cuda.synchronize()
                wall_time = time.time() - start_time + wall_time
                allTarget.append(target)
                #allOutput.append(output)

                #得画一个输出概率和正确性关系的图像.        
                #measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                        
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
                # measure elapsed time

                if (i+1) % (1+(self.args.print_freq//output.shape[0])) == 0:
                    progress.display(i)

            progress.display_summary()

            self.logger.info(f"Time = {wall_time:.2f}")

            self.logger.info(f"top1:{top1.avg}")

            self.logger.info(f"top5:{top5.avg}")

            allTarget=torch.cat(allTarget)
            allOutput=torch.cat(allOutput)

            self.logger.info((f" > domain is {self.args.stream.corruptions[idx]}"))

            if self.args.stream.num_class<=10:
                utils.classwise_stats(allOutput.cpu(),allTarget.cpu(),self.logger)

            self.logger.info(f"Target shape:{allTarget.shape}")
            self.logger.info(f"Output shape:{allOutput.shape}")
            self.reset()

    def reset(self):

        self.mem=PBRS(self.args.method.queue_size,self.args.stream.num_class,updateCriterion=self.args.method.updateCriterion)
        self.tptmodel.reset()

        for name, param in self.tptmodel.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        self.originmodel=deepcopy(self.tptmodel)

        self.optimizer.load_state_dict(self.optim_state) 


def setup(model, args,logger,dataset):
    logger.info("Setup TTA method: ours")
    ours_model = ours(model,args,logger)
    logger.info(f"model for adaptation: %s", ours_model)
    ours_model.reset()
    testOursmodel=FixtestData(ours_model,dataset,logger,args)#(tpt的模型,数据集的名称,打印的logger,参数列表args)

    return testOursmodel
