from numpy import ndarray
from pandas import DataFrame, Series
from typing import Iterable, Any, Union

import pandas as pd


class Dataset:

    # Unlabelled and Labelled datasets are quite similar, but slightly different
    # This base class encapsulates the common functionality of both
    class BaseDataset:

        def __init__(self, data: DataFrame, parent: 'Dataset', linked: bool = False):

            self._data = data
            self._parent = parent

            self.create_child = lambda child_data: parent.create_dataset(child_data, parent, True)

            if not linked:
                self._data = data.copy()
                self._data.set_index(self._data.columns.values[0], inplace=True)


            # By convention, the last column is the label.
            # We want to call this column 'label'
            self._data.rename(columns={list(self._data)[-1]: 'label'}, inplace=True)
            self._label_column = self._data.columns[-1]

            # By convention, all other columns are features
            self._feature_columns = self._data.columns[0:-1]

            if not linked:
                # We want to treat labels as numeric. If labels are strings we need
                # to create a lookup to map from strings.
                self._label_lookup = factorize_labels(self._data[self._label_column])
                self._data[self._label_column] = self._data[self._label_column].map(self._label_lookup)

        @property
        def ids(self):
            return self._data.index.values

        @property
        def features(self):
            return self._data[self._feature_columns]

        @features.setter
        def features(self, new_features):
            self._data[self._feature_columns] = new_features

        @property
        def labels(self) -> ndarray:
            return self._data[self._label_column].values

        @labels.setter
        def labels(self, new_labels):
            self._data[self._label_column] = new_labels

        @property
        def shape(self):
            return self._data.shape

        @property
        def row_count(self):
            return self._data.shape[0]

        @property
        def col_count(self):
            return self._data.shape[1]

        @property
        def index(self):
            return self._data.index

        def head(self, n: int = 5):
            return self.create_child(self._data.head(n))

        def by_id(self, row_id: Union[int, Iterable[int]]) -> 'Dataset':
            # need to remove any rows which aren't in the index
            values = self._data.loc[row_id,:]
            if isinstance(values, Series):
                values = values.to_frame().transpose()
            return self.create_child(pd.concat([pd.DataFrame(), values]))

        def get_labels(self, data: DataFrame):
            return data[self._label_column]

        def drop(self, ids):
            return self.create_child(self._data.drop(ids))

        def append(self, rows_to_append: 'Dataset'):
            return self.create_child(self._data.append(rows_to_append._data))

        def copy(self):
            return self._data.copy()

        def sample(self, n: int, random_state: int):
            return self.create_child(self._data.sample(n=n, random_state=random_state))

    def __init__(self, data: DataFrame):

        # All data is unlabelled to begin with
        self._unlabelled = self.BaseDataset(data, self)

        # Create an empty data frame to hold our labelled data
        empty_data_frame = DataFrame().reindex_like(data).iloc[0:0]
        self._labelled = self.BaseDataset(empty_data_frame, self)

    @property
    def labelled(self):
        return self._labelled

    @property
    def unlabelled(self):
        return self._unlabelled

    @property
    def features(self):
        return pd.concat([self.unlabelled.features, self.labelled.features])

    def add_labels(self, ids: Iterable[Any], labels: Iterable[Any]):
        rows_to_label = self._unlabelled.by_id(ids)
        rows_to_label.labels = labels
        self._unlabelled = self._unlabelled.drop(ids)
        self._labelled = self._labelled.append(rows_to_label)

    def create_dataset(self, data: DataFrame, parent: 'Dataset', linked: bool = False):
        return self.BaseDataset(data, parent, linked)

    def by_id(self, ids):
        unlabelled_matches = self.unlabelled.by_id(ids)._data
        labelled_matches = self.labelled.by_id(ids)._data
        all_matches = pd.concat([labelled_matches, unlabelled_matches])
        # need to re-order to match ids

        return self.create_dataset(all_matches, False)

def factorize_labels(labels):
    return {name: index for index, name in enumerate(labels.unique())}
