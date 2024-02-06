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

from data.datautils import AugMixAugmenter, build_dataset

from data.fewshot_datasets import fewshot_datasets

import numpy as np

from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight, set_random_seed
from data.cls_to_names import *

from data.imagenet_ar import A,V, R,ImageNetdata

from data.corruption_datasets import corruptions_datasets
from data.imagnet_prompts import imagenet_classes

__all__ = ["setup"]

class tpt(nn.Module):
    """Norm adapts a model by estimating feature statistics during testing.

    Once equipped with Norm, the model normalizes its features during testing with batch-wise statistics, just like batch norm does during training.
    """

    def __init__(self, model,args,logger):
        super().__init__()
        self.model = model
        self.args=args.method
        
        for name, param in self.model.named_parameters():
            if not args.cocoop:
                if "prompt_learner" not in name:
                    param.requires_grad_(False)
            else:
                if "text_encoder" not in name:
                    param.requires_grad_(False)

        logger.info(f"=> Model created: visual backbone {args.arch}")
    
        if not torch.cuda.is_available():
            logger.info('using CPU, this will be slow')
        else:
            assert args.gpu is not None
            torch.cuda.set_device(args.gpu)
            self.model = self.model.cuda(args.gpu)
        
        trainable_param = self.model.prompt_learner.parameters()
        self.optimizer = torch.optim.AdamW(trainable_param, self.args.optim.lr)
        self.optim_state = deepcopy(self.optimizer.state_dict())

        self.scaler = torch.cuda.amp.GradScaler(init_scale=1000)

        cudnn.benchmark = True

        self.model_state = deepcopy(self.model.state_dict())
        #deepcopy()进行深拷贝，防止父对象和子对象公用同一个地址

    def forward(self, x):
        return self.model(x)
    
    def test_time_tuning(self,x):
        self.optimizer.load_state_dict(self.optim_state)
        for j in range(self.args.tta_steps):
            with torch.cuda.amp.autocast():
                output = self.model(x)
                #print("output:",output.shape) 
                output, selected_idx = self.select_confident_samples(output, self.args.selection_p)

                loss = self.avg_entropy(output)
        
            self.optimizer.zero_grad()
            # compute gradient and do SGD step
            self.scaler.scale(loss).backward()
            # Unscales the gradients of optimizer's assigned params in-place
            self.scaler.step(self.optimizer)
            self.scaler.update()

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optim_state)

    def select_confident_samples(self,logits, top):
        #根据logits选最置信的前6个进行输出
        batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
        idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
        return logits[idx], idx

    def avg_entropy(self,outputs):
        logits = outputs - outputs.logsumexp(dim=-1, keepdim=True) # logits = outputs.log_softmax(dim=1) [N, 1000]
        avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0]) # avg_logits = logits.mean(0) [1, 1000]
        min_real = torch.finfo(avg_logits.dtype).min #获取对应精度的最小值
        avg_logits = torch.clamp(avg_logits, min=min_real)
        return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)

class testData():
    def __init__(self, tptmodel,dataset,logger,args):
        self.tptmodel=tptmodel
        self.dataset=dataset
        self.logger=logger
        self.args=args

        normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])
    #不清楚这个normalize在换成cifar相关的数据集后需不需要做一些修改

        base_transform = transforms.Compose([
                transforms.Resize(args.resolution, interpolation=BICUBIC),
                transforms.CenterCrop(args.resolution)])
        preprocess = transforms.Compose([
                transforms.ToTensor(),
                normalize])
        data_transform = AugMixAugmenter(base_transform, preprocess, n_views=args.batch_size-1, 
                                            augmix=len(dataset)>1)
        batchsize = 1

        if dataset in fewshot_datasets:
            classnames = eval("{}_classes".format(dataset.lower()))
        elif args.test_sets in corruptions_datasets:
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
            
            wall_time=0
            for i, (images, target) in enumerate(val_loader):
                #print("some tips:",images.shape,target.shape)
                if isinstance(images, list):
                    for k in range(len(images)):
                        images[k] = images[k].cuda(self.args.gpu, non_blocking=True)
                    image = images[0]
                else:
                    if len(images.size()) > 4:
                        # when using ImageNet Sampler as the dataset
                        assert images.size()[0] == 1
                        images = images.squeeze(0)
                    images = images.cuda(self.args.gpu, non_blocking=True)
                    image = images
                target = target.cuda(self.args.gpu, non_blocking=True)
                
                images = torch.cat(images, dim=0)

                # reset the tunable prompt to its initial state
                torch.cuda.synchronize()
                start_time = time.time()
                if self.args.method.tta_steps > 0 and self.args.tptContinual==False:
                    #print("123321")
                    with torch.no_grad():
                        self.tptmodel.reset()
                    

                #记着这里的优化器需要重新恢复之前的状态
                # optimizer.load_state_dict(optim_state)
                # test_time_tuning(model, images, optimizer, scaler, args)
                self.tptmodel.test_time_tuning(images)
                
                # The actual inference goes here
                
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        output = self.tptmodel(image)
                        # print("image:",type(image))
                        # print("image:",image.shape)
                # measure accuracy and record loss
                torch.cuda.synchronize()
                wall_time = time.time() - start_time + wall_time
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                        
                top1.update(acc1[0], image.size(0))
                top5.update(acc5[0], image.size(0))
                # measure elapsed time

                if (i+1) % self.args.print_freq == 0:
                    progress.display(i)

            progress.display_summary()

            self.logger.info(f"Time = {wall_time:.2f}")

            self.logger.info(f"top1:{top1.avg}")

            self.logger.info(f"top5:{top5.avg}")


def setup(model, args,logger,dataset):
    logger.info("Setup TTA method: tpt")
    tpt_model = tpt(model,args,logger)
    logger.info(f"model for adaptation: %s", tpt_model)
    tpt_model.reset()
    testTptmodel=testData(tpt_model,dataset,logger,args)#(tpt的模型,数据集的名称,打印的logger,参数列表args)

    return testTptmodel
