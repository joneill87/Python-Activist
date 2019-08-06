from Activist.dataset import Dataset
from Activist.seeders.seedingstrategy import SeedingStrategy
from Activist.selectionstrategy import SelectionStrategy

from typing import Iterable


class LabellingJob:

    _data: Dataset
    _seeding_strategy: SeedingStrategy
    _seed_size: int
    _selection_strategy: SelectionStrategy
    _batch_size: int

    _next_query_ids: Iterable[int]
    _batch_number = 0

    def __init__(self,
                 data: Dataset,
                 seeding_strategy: SeedingStrategy,
                 seed_size: int,
                 selection_strategy: SelectionStrategy,
                 batch_size: int):
        self._data = data
        self._seeding_strategy = seeding_strategy
        self._seed_size = seed_size
        self._selection_strategy = selection_strategy
        self._batch_size = batch_size

    def get_next_query(self):
        return self._data



