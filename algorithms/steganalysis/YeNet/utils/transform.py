import numpy as np
import torch
import random

class ToNumpy(object):
    """Преобразует PIL Image в numpy array"""
    def __call__(self, pic):
        return np.array(pic)

class RandomRot(object):
    def __call__(self, sample):
        rot = random.randint(0, 3)
        
        # Проверяем размерность массива
        if sample.ndim == 2:
            # Для 2D изображения (H, W)
            return np.rot90(sample, rot).copy()
        elif sample.ndim == 3:
            # Для 3D изображения (C, H, W) или (H, W, C)
            # Предполагаем, что оси для вращения - последние две (H, W)
            return np.rot90(sample, rot, axes=(-2, -1)).copy()
        else:
            raise ValueError(f"Unexpected array dimension: {sample.ndim}")

class RandomFlip(object):
    def __init__(self, p=0.5):
        self._p = p

    def __call__(self, sample):
        if random.random() < self._p:
            if sample.ndim == 2:
                # Для 2D изображения (H, W) - отражаем по ширине (ось 1)
                return np.flip(sample, axis=1).copy()
            elif sample.ndim == 3:
                # Для 3D изображения (C, H, W) - отражаем по ширине (ось 2)
                # Или для (H, W, C) - отражаем по ширине (ось 1)
                # Предполагаем формат (C, H, W)
                return np.flip(sample, axis=2).copy()
            else:
                raise ValueError(f"Unexpected array dimension: {sample.ndim}")
        else:
            return sample
        

class ToNumpyList:
    def __call__(self, images):
        return [np.array(img) for img in images]

class RandomRotList:
    def __call__(self, images):
        rot = random.randint(0, 3)
        # Rotate all images using the same rot, assuming images are numpy arrays (H,W) or (H,W,C)
        return [np.rot90(img, rot, axes=(0, 1)).copy() for img in images]

class RandomFlipList:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, images):
        if random.random() < self.p:
            # Flip horizontally (axis=1, which is width)
            return [np.flip(img, axis=1).copy() for img in images]
        else:
            return images

class ToTensorList:
    def __call__(self, images):
        result = []
        for img in images:
            if img.ndim == 2:
                # grayscale: (H,W) -> (1,H,W)
                tensor = torch.from_numpy(img).unsqueeze(0).float()
            elif img.ndim == 3:
                # RGB: (H,W,C) -> (C,H,W)
                tensor = torch.from_numpy(img).permute(2,0,1).float()
            else:
                raise ValueError(f"Unexpected array dimensions: {img.ndim}")
            result.append(tensor)
        return result
