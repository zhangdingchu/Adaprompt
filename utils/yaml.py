from argparse import Namespace
from genericpath import isfile
import os
import yaml

__all__ = ["config2args"]

def config2args(args, path):#将path中的所有字典属性设置到namespace中
    assert isinstance(path, str) and os.path.isfile(path), f"Config file {path} not found. "
    with open(path, 'r', encoding='utf-8') as file: 
        raw = yaml.safe_load(file)#返回字典
    
    #print("raw:",raw)

    def helper(raw, args):
        for key in raw:
            if isinstance(raw[key], dict):
                ret = helper(raw[key], Namespace())
                setattr(args, key, ret)
            else:
                setattr(args, key, raw[key])
        return args
    return helper(raw, args)
