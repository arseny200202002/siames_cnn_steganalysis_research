import os
import numpy as np
import re
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from torch.autograd import Variable


class StegoDataset(Dataset):
    """Датасет для стегоанализа с фильтрацией по параметрам"""
    
    def __init__(self, cover_dir, stego_dir, 
                 bpp=None, 
                 algorithm_name=None, 
                 resize_strategy=None, 
                 W=None, 
                 H=None,
                 num_samples=None, 
                 transform=None):
        """
        Args:
            cover_dir: путь к папке с cover изображениями (label=0)
            stego_dir: путь к папке со стегоизображениями (label=1)
            bpp: бит на пиксель (например, 0.1, 0.2, 0.4)
            algorithm_name: название алгоритма (например, 'hill', 'wow', 'suniward')
            resize_strategy: стратегия изменения размера ('resize', 'center_crop', 'random_crop')
            W: ширина изображения
            H: высота изображения
            num_samples: количество пар изображений (если None - все)
            transform: трансформации для изображений
        """
        self.cover_dir = cover_dir
        self.stego_dir = stego_dir
        self.transform = transform
        
        # Сохраняем параметры фильтрации
        self.bpp = bpp
        self.algorithm_name = algorithm_name
        self.resize_strategy = resize_strategy
        self.W = W
        self.H = H
        
        # Получаем все stego файлы
        all_stego_files = [f for f in os.listdir(stego_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.pgm'))]
        
        # Фильтруем stego файлы по заданным параметрам
        stego_files = self._filter_stego_files(all_stego_files)
        
        print(f"Найдено {len(stego_files)} stego файлов по заданным параметрам")
        
        # Ограничиваем количество, если нужно
        if num_samples is not None and num_samples < len(stego_files):
            stego_files = stego_files[:num_samples]
        
        # Создаем список файлов
        self.files = []
        matched_cover_count = 0
        
        # Для каждого stego файла находим соответствующий cover
        for stego_file in stego_files:
            # Извлекаем оригинальное имя файла из stego_файла
            original_filename = self._extract_original_filename(stego_file)
            
            if original_filename:
                # Ищем cover файл с таким же именем
                cover_path = self._find_cover_file(original_filename)
                
                if cover_path:
                    # Добавляем пару в датасет
                    self.files.append({
                        'cover_path': cover_path,
                        'stego_path': os.path.join(stego_dir, stego_file),
                        'label': 1  # Для stego метка 1
                    })
                    matched_cover_count += 1
                    
                    # Добавляем cover как отдельный образец с меткой 0
                    self.files.append({
                        'cover_path': cover_path,
                        'stego_path': None,
                        'label': 0
                    })
        
        print(f"Загружено {len(self.files)} изображений ({len(stego_files)} пар)")
        
    def _filter_stego_files(self, files):
        """Фильтрует stego файлы по заданным параметрам"""
        filtered = []
        
        # Создаем шаблон для поиска в имени файла
        pattern_parts = []
        
        if self.resize_strategy is not None:
            pattern_parts.append(re.escape(str(self.resize_strategy)))
        else:
            pattern_parts.append(r'[^_]+')  # любое значение
        
        if self.W is not None:
            pattern_parts.append(re.escape(str(self.W)))
        else:
            pattern_parts.append(r'\d+')
            
        if self.H is not None:
            pattern_parts.append(re.escape(str(self.H)))
        else:
            pattern_parts.append(r'\d+')
            
        if self.algorithm_name is not None:
            pattern_parts.append(re.escape(str(self.algorithm_name)))
        else:
            pattern_parts.append(r'[^_]+')
            
        if self.bpp is not None:
            # Для bpp может быть с точкой, например 0.1bpp
            bpp_str = str(self.bpp).replace('.', '\.')
            pattern_parts.append(f'{bpp_str}bpp')
        else:
            pattern_parts.append(r'[\d\.]+bpp')
        
        # Добавляем оставшуюся часть имени файла
        pattern_parts.append(r'.+')
        
        # Собираем полный паттерн
        pattern = '^' + '_'.join(pattern_parts) + '$'
                
        # Фильтруем файлы
        for f in files:
            if re.match(pattern, f):
                filtered.append(f)
        
        return filtered
    
    def _extract_original_filename(self, stego_filename):
        """
        Извлекает оригинальное имя файла из stego_файла.
        Формат: resize_strategy_W_H_algorithm_name_bpp_filename
        """
        try:
            # Разделяем по '_'
            parts = stego_filename.split('_')
            
            # Определяем, где начинается оригинальное имя файла
            # После bpp (которое содержит 'bpp')
            bpp_index = -1
            for i, part in enumerate(parts):
                if 'bpp' in part:
                    bpp_index = i
                    break
            
            if bpp_index != -1 and bpp_index + 1 < len(parts):
                # Оригинальное имя - это всё, что после bpp
                original_parts = parts[bpp_index + 1:]
                original_filename = '_'.join(original_parts)
                return original_filename
            else:
                print(f"Не удалось извлечь имя из: {stego_filename}")
                return None
        except Exception as e:
            print(f"Ошибка при извлечении имени из {stego_filename}: {e}")
            return None
    
    def _find_cover_file(self, filename):
        """Ищет cover файл по имени"""
        # Пробуем разные варианты расширений
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.pgm']:
            # Проверяем, есть ли уже расширение в имени
            if '.' in filename:
                # Если расширение уже есть, используем как есть
                cover_filename = f'{self.resize_strategy}_{self.W}_{self.H}_{filename}'
                cover_path = os.path.join(self.cover_dir, cover_filename)
                if os.path.exists(cover_path):
                    return cover_path
            else:
                # Если нет расширения, пробуем добавить
                cover_filename = f'{self.resize_strategy}_{self.W}_{self.H}_{filename}'
                cover_path = os.path.join(self.cover_dir, cover_filename + ext)
                if os.path.exists(cover_path):
                    return cover_path
        
        return None
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        item = self.files[idx]
        
        # Загружаем изображение
        if item['label'] == 0:  # cover
            image_path = item['cover_path']
        else:  # stego
            image_path = item['stego_path']
        
        # Загружаем изображение
        image = Image.open(image_path).convert('L')

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(item['label'], dtype=torch.long)
        sample = {'images': image, 'labels': label}
        
        return sample
