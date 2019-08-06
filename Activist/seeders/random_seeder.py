from typing import Iterable
import random

from Activist.dataset import Dataset
from Activist.seeders.seedingstrategy import SeedingStrategy


class RandomSeeder(SeedingStrategy):

    def __init__(self, random_seed: int):
        random.seed(random_seed)

    def get_seed_data(self, data: Dataset, n: int) -> Iterable[int]:
        random_indices = random.sample(range(data.unlabelled.row_count), n)
        return data.unlabelled.by_index(random_indices).ids

