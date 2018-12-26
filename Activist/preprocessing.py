from abc import abstractmethod
from dataset import Dataset
from typing import List


class PreprocessingStep:

    @abstractmethod
    def process_step(self, data: Dataset) -> Dataset:
        pass


class PreprocessingQueue:

    def __init__(self, preprocessing_steps: List[PreprocessingStep]):
        # if a single step has been passed coerce it to a list
        if isinstance(preprocessing_steps, list):
            self.steps = preprocessing_steps
        else:
            self.steps = [preprocessing_steps]

    def preprocess_data(self, data: Dataset) -> Dataset:
        for step in self.steps:
            data = step.process_step(data)
        return data



