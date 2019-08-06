from abc import abstractmethod
from typing import Iterable

from Activist.dataset import Dataset


class SeedingStrategy:

    def __init__(self):
        pass

    @abstractmethod
    def get_seed_data(self, data: Dataset, n: int) -> Iterable[int]:
        pass

