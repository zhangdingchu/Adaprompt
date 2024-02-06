from argparse import Namespace
import os
import sys
import logging
import numpy as np

__all__ = [
    "set_logger", "log_args", "print_args", 
    "classwise_stats"
]

def set_logger(log_path, log_name):#根据path和name进行日志的保存并输出到控制台
    if not os.path.isdir(log_path): os.makedirs(log_path)#进行目录的创建

    #例子1开始

    # logger = logging.getLogger("logger")
    # handler = logging.StreamHandler()
    # handler.setLevel(logging.INFO)
    # formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    # handler.setFormatter(formatter)
    # logger.addHandler(handler)
    # logger.debug('This is a customer debug message')
    # logging.info('This is an customer info message')
    # logger.warning('This is a customer warning message')
    # logger.error('This is an customer error message')
    # logger.critical('This is a customer critical message')

    #例子1结束

    #例子2开始
    logger = logging.getLogger("logger")
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(filename=os.path.join(log_path, log_name + ".txt"),mode="w")
    logger.setLevel(logging.INFO)
    handler1.setLevel(logging.INFO)
    handler2.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger.addHandler(handler1)
    logger.addHandler(handler2)

    return logger
    # logger.debug('This is a customer debug message')
    # logger.info('This is an customer info message')
    # logger.warning('This is a customer warning message')
    # logger.error('This is an customer error message')
    # logger.critical('This is a customer critical message')
    #例子2结束


    # logging.basicConfig(level=logging.INFO,
    #     format='%(asctime)s %(name)s %(levelname)s %(message)s',
    #     datefmt='%m/%d/%Y %I:%M:%S %p',
    #     handlers=[
    #         logging.FileHandler(os.path.join("./"+log_path, log_name + ".txt"), mode='w'),
    #         logging.StreamHandler(sys.stdout)
    #     ]
    # )
    #logging.basicConfig(level=logging.INFO)


def log_args(args,logger):
    def helper(args,logger, prefix=""):
        for arg in vars(args):
            content = getattr(args, arg)
            if isinstance(content, Namespace): 
                helper(content,logger, prefix+arg+".")
            else:
                logger.info(f" > {prefix}{arg} = {content}")
    assert isinstance(args, Namespace)
    logger.info("Experiment args: ")
    helper(args,logger)

def print_args(args):
    def helper(args, prefix=""):
        for arg in vars(args):
            content = getattr(args, arg)
            if isinstance(content, Namespace): 
                helper(content, prefix+arg+".")
            else:
                print(f" > {prefix}{arg} = {content}")
    assert isinstance(args, Namespace)
    print("Experiment args: ")
    helper(args)

 
def classwise_stats(preds, gts,logger):
    logger.info("Classwise Stats")
    preds = preds.max(1)[1].numpy()
    gts = gts.numpy()
    #print("shape:",preds.shape,gts.shape)

    num_classes = int(preds.max() + 1)
    #print("numclass:",num_classes)
    logger.info(" > Clas Idx" + "\t".join([f"{i:6d}" for i in range(num_classes)]))

    cnts = [np.sum(gts == i) for i in range(num_classes)]
    logger.info(" > Gts  Cnt" + "\t".join([f"{cnts[i]:6d}" for i in range(num_classes)]))

    cnts = [np.sum(preds == i) for i in range(num_classes)]
    logger.info(" > Pred Cnt" + "\t".join([f"{cnts[i]:6d}" for i in range(num_classes)]))
    
    corr = preds == gts
    accs = [np.sum(corr[gts == i]) / np.sum(gts == i) for i in range(num_classes)]
    logger.info(" > Clas Acc" + "\t".join([f"{accs[i]:2.2%}" for i in range(num_classes)]))

    #print("opensetAcc:",accs[len(accs)-1]*4000/cnts[len(cnts)-1])
    