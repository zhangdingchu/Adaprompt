from copy import deepcopy

import torch
import torch.nn as nn

import torch.backends.cudnn as cudnn
import time
import torchvision.transforms as transforms

from PIL import Image

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from data.datautils import AugMixAugmenter,BaseTransform, build_dataset

from data.fewshot_datasets import fewshot_datasets
from data.imagnet_prompts import imagenet_classes
from data.corruption_datasets import corruptions_datasets
from data.imagenet_ar import A, R,V,ImageNetdata
import numpy as np

from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight, set_random_seed
from data.cls_to_names import *

import utils


__all__ = ["setup"]

class source(nn.Module):
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

    def forward(self, x):
        return self.model(x)
    
    def test_time_tuning(self,x):
        pass

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)

class testData():
    def __init__(self, tptmodel,dataset,logger,args):
        self.tptmodel=tptmodel
        self.dataset=dataset
        self.logger=logger
        self.args=args

        normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

        base_transform = transforms.Compose([
                transforms.Resize(args.resolution, interpolation=BICUBIC),
                transforms.CenterCrop(args.resolution)])
        preprocess = transforms.Compose([
                transforms.ToTensor(),
                normalize])
        data_transform = BaseTransform(base_transform, preprocess)

        batchsize = 64

        if dataset in fewshot_datasets:
            classnames = eval("{}_classes".format(dataset.lower()))
        #lower函数:将字符串中的所有大写改成小写
        #classnames=caltech101_classes
        elif dataset in corruptions_datasets:
            classnames=eval("{}_classes".format(dataset))
        elif dataset in ImageNetdata:
            classnames=eval("{}".format(dataset))
        else:
            classnames = imagenet_classes

        # if dataset in fewshot_datasets:
        #     classnames = eval("{}_classes".format(dataset.lower()))
        # else:
        #     classnames =eval("{}_classes".format(dataset))

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
            with torch.no_grad():
                self.tptmodel.reset()

            allOutput=[]
            allTarget=[]
            
            wall_time=0
            for i, (images, target) in enumerate(val_loader):
                #print("some tips:",images.shape,target.shape)
                #记着这里的优化器需要重新恢复之前的状态
                # optimizer.load_state_dict(optim_state)
                # test_time_tuning(model, images, optimizer, scaler, args)
                images=images.cuda(self.args.gpu, non_blocking=True)
                target=target.cuda(self.args.gpu, non_blocking=True)
                # The actual inference goes here
                #print(target)
                torch.cuda.synchronize()
                start_time = time.time()
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        output = self.tptmodel(images)
                torch.cuda.synchronize()
                wall_time = time.time() - start_time + wall_time
                # measure accuracy and record loss
                allTarget.append(target)
                allOutput.append(output)

                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                #print(f"{output} and {target} and {acc1} and {acc5}")
                        
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
                # measure elapsed time

                if (i+1) % (self.args.print_freq) == 0:
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


def setup(model, args,logger,dataset):
    logger.info("Setup TTA method: Source")
    source_model = source(model,args,logger)
    logger.info(f"model for adaptation: %s", source_model)
    source_model.reset()
    testSourcemodel=testData(source_model,dataset,logger,args)#(tpt的模型,数据集的名称,打印的logger,参数列表args)

    return testSourcemodel
