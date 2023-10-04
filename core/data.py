import pickle
from dataclasses import dataclass, field

import numpy as np

from .alias import Array
from .errors import LoadError


@dataclass
class Data:
    """Сырые данные, полученные со спектрометра"""
    intensity: Array[float]  # Двумерный массив данных измерения. Первый индекс - номер кадра, второй - номер сэмпла в кадре
    clipped: Array[bool]  # Двумерный массив boolean значений. Если `clipped[i,j]==True`, то `intensity[i,j]` содержит зашкаленное значение
    tau: int  # Экспозиция в миллисекундах

    def __post_init__(self):
        self._time = self._time = np.arange(self.n_times)
        self._number = np.arange(self.n_numbers)

    @property
    def n_times(self) -> int:
        """Количество измерений"""
        return self.intensity.shape[0]

    @property
    def time(self) -> Array[int]:
        return self._time

    @property
    def n_numbers(self) -> int:
        """Количество отсчетов"""
        return self.intensity.shape[1]

    @property
    def number(self) -> Array[int]:
        return self._number

    @property
    def shape(self) -> tuple[int, int]:
        """Размерность данынх"""
        return self.intensity.shape

    def save(self, path: str):
        """Сохранить объект в файл"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> 'Data':
        """Прочитать объект из файла"""

        with open(path, 'rb') as f:
            result = pickle.load(f)

        if not isinstance(result, cls):
            raise LoadError(path)

        return result

    def __repr__(self) -> str:
        cls = self.__class__
        return f'{cls.__name__}({self.n_times=}, {self.n_numbers=})'


@dataclass
class Spectrum(Data):
    """Обработанные данные, полученные со спектрометра.
    Содержит в себе информацию о длинах волн измерения.
    В данный момент обработка заключается в вычитании темнового сигнала.
    """

    wavelength: Array[float]  # длина волны

    def __repr__(self) -> str:
        cls = self.__class__
        return f'{cls.__name__}({self.n_times=}, {self.n_numbers=})'
