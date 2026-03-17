import torch
import shutil

def save_checkpoint(state, is_best, filename=r'checkpoints\checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, r'checkpoints\model_best.pth.tar')