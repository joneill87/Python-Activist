import math
import pandas as pd
import numpy as np

class SelectionStrategy:

    def get_next_query(self, data, num_obs):
        pass


class RandomSelectionStrategy(SelectionStrategy):

    def __init__(self, random_seed):
        self.random_seed = random_seed

    def get_next_query(self, data, num_obs):
        next_query = data.sample(n=num_obs, random_state=self.random_seed).id
        return next_query


class ConfidenceSelectionStrategy(SelectionStrategy):

    def __init__(self):
        pass

    def get_next_query(self, data, num_obs):
        return data.sort_values('confidence').head(num_obs)['id']


class HedgedSelectionStrategy(SelectionStrategy):

    def __init__(self, confidenceRatio):
        if confidenceRatio > 1:
            raise TypeError("confidenceRatio parameter must be <= 1")
        self.confidenceRatio = confidenceRatio

    def get_next_query(self, data, num_obs):
        confident_obs = math.floor(num_obs * self.confidenceRatio)
        uncertain_obs = num_obs - confident_obs
        return data.sort_values('confidence').iloc[np.r_[0:confident_obs, (uncertain_obs * -1):0]]['id']

