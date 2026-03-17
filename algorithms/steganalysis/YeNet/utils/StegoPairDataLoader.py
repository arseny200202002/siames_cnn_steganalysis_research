import torch
from torch.utils.data import DataLoader

class StegoPairDataLoader(DataLoader):
    """
        Кастомный DataLoader с перемешиванием cover и stego внутри батча.
        Наследуется от torch.utils.data.DataLoader.
    """
    
    def __init__(
            self, 
            dataset, 
            batch_size=32, 
            shuffle=True, 
            num_workers=4, 
            pin_memory=False, 
            drop_last=False, 
            **kwargs
        ):
        if batch_size % 2 != 0:
            raise ValueError(f'batch_size should be even, got: {batch_size}')
        super(StegoPairDataLoader, self).__init__(
            dataset=dataset,
            batch_size=int(batch_size/2),
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            **kwargs
        )
        
    def __iter__(self):
        iterator = super(StegoPairDataLoader, self).__iter__()
        return self._generator(iterator)
    
    def _generator(self, iterator):
        for batch_data in iterator:
            cover_batch = batch_data['cover']
            stego_batch = batch_data['stego']
            
            batch_size = cover_batch.size(0)
            
            # Объединяем
            all_images = torch.cat([cover_batch, stego_batch], dim=0)
            all_labels = torch.cat([
                torch.zeros(batch_size, dtype=torch.long),
                torch.ones(batch_size, dtype=torch.long)
            ])
            
            # Перемешиваем внутри батча
            perm = torch.randperm(batch_size * 2)
            images = all_images[perm]
            labels = all_labels[perm]
            
            yield {
                "images": images,
                "labels": labels,
            }