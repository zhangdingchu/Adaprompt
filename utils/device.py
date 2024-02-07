
import torch
import torch.backends.cudnn as cudnn
import random
import numpy as np

__all__ = ["set_device"]

def set_device(seed):#设置随机种子seed导致每次运行的结果相同,有cuda返回cuda否则返回cpu，返回值为字符串
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cudnn.benchmark = True
    torch.manual_seed(seed)#设置随机种子导致每次运行的结果相同
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    return device