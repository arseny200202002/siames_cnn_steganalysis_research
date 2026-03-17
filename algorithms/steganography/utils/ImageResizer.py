from PIL import Image
import random
from typing import Union, Tuple, Optional

class ImageResizer:
    """
    Модуль для изменения размера изображений с различными стратегиями.
    
    Поддерживаемые стратегии:
    - 'resize' - масштабирование изображения до целевого размера
    - 'center_crop' - вырезание центральной области целевого размера
    - 'random_crop' - вырезание случайной области целевого размера
    """
    
    def __init__(self, target_size: Union[int, Tuple[int, int]], strategy: str = 'resize'):
        """
        Args:
            target_size: Целевой размер. Может быть:
                        - int: (target_size, target_size) для квадрата
                        - tuple: (width, height)
            strategy: Стратегия изменения размера ('resize', 'center_crop', 'random_crop')
        """
        self.target_size = self._parse_size(target_size)
        self.strategy = strategy.lower()
        
        if self.strategy not in ['resize', 'center_crop', 'random_crop']:
            raise ValueError(f"Неизвестная стратегия: {strategy}. "
                           f"Доступны: 'resize', 'center_crop', 'random_crop'")
    
    def _parse_size(self, size: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
        """Преобразует входной размер в кортеж (width, height)"""
        if isinstance(size, int):
            return (size, size)
        elif isinstance(size, (tuple, list)) and len(size) == 2:
            return (int(size[0]), int(size[1]))
        else:
            raise ValueError(f"Некорректный размер: {size}. Ожидается int или tuple (width, height)")
    
    def _get_crop_coords(self, image: Image.Image, 
                         crop_width: int, crop_height: int, 
                         random_crop: bool = False) -> Tuple[int, int, int, int]:
        """
        Вычисляет координаты для вырезания области
        
        Args:
            image: Исходное изображение
            crop_width: Ширина вырезаемой области
            crop_height: Высота вырезаемой области
            random_crop: Если True - случайное место, если False - центр
            
        Returns:
            Кортеж (left, top, right, bottom)
        """
        img_width, img_height = image.size
        
        # Проверяем, что изображение достаточно большое для вырезания
        if img_width < crop_width or img_height < crop_height:
            raise ValueError(f"Изображение ({img_width}x{img_height}) меньше целевого размера "
                           f"({crop_width}x{crop_height}) для обрезки")
        
        if random_crop:
            # Случайная позиция
            left = random.randint(0, img_width - crop_width)
            top = random.randint(0, img_height - crop_height)
        else:
            # Центральная позиция
            left = (img_width - crop_width) // 2
            top = (img_height - crop_height) // 2
        
        right = left + crop_width
        bottom = top + crop_height
        
        return (left, top, right, bottom)
    
    def resize(self, image: Image.Image) -> Image.Image:
        """
        Масштабирует изображение до целевого размера
        
        Args:
            image: PIL Image объект
            
        Returns:
            Измененное PIL Image
        """
        return image.resize(self.target_size, Image.Resampling.LANCZOS)
    
    def center_crop(self, image: Image.Image) -> Image.Image:
        """
        Вырезает центральную область целевого размера
        
        Args:
            image: PIL Image объект
            
        Returns:
            Обрезанное PIL Image
        """
        crop_coords = self._get_crop_coords(image, self.target_size[0], self.target_size[1], 
                                           random_crop=False)
        return image.crop(crop_coords)
    
    def random_crop(self, image: Image.Image) -> Image.Image:
        """
        Вырезает случайную область целевого размера
        
        Args:
            image: PIL Image объект
            
        Returns:
            Обрезанное PIL Image
        """
        crop_coords = self._get_crop_coords(image, self.target_size[0], self.target_size[1], 
                                           random_crop=True)
        return image.crop(crop_coords)
    
    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Применяет выбранную стратегию к изображению
        
        Args:
            image: PIL Image объект
            
        Returns:
            Обработанное PIL Image
        """
        if self.strategy == 'resize':
            return self.resize(image)
        elif self.strategy == 'center_crop':
            return self.center_crop(image)
        elif self.strategy == 'random_crop':
            return self.random_crop(image)
        else:
            # Эта ошибка не должна возникнуть из-за проверки в __init__
            raise ValueError(f"Неизвестная стратегия: {self.strategy}")


