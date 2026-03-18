import torch
import shutil
from os.path import join

def save_checkpoint(state, is_best, filename=join('checkpoints', 'checkpoint.pth.tar')):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(
            filename, 
            join('checkpoints', 'model_best.pth.tar')
        )