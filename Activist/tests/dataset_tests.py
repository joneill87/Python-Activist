import unittest

from dataset import Dataset

import pandas as pd


class TestDatasetMethods(unittest.TestCase):

    def setUp(self):
        self.data = pd.DataFrame([
            ['id_1', 'one', 'ein', 'label_1'],
            ['id_2', 'two', 'zwei', 'label_2'],
            ['id_3', 'three', 'drei', 'label_3'],
            ['id_4', 'four', 'vier', 'label_4']
                                ],
                                 columns=['id', 'f1', 'f2', 'label'])

        self.dataset = Dataset(self.data)

    def test_labelled_and_unlabelled_created(self):
        self.assertIsNotNone(self.dataset.labelled, "The Labelled dataset should not be null")
        self.assertIsNotNone(self.dataset.unlabelled, "The unlabelled dataset should not be null")

    def test_unlabelled_rowcount_equals_dataframe_rowcount(self):
        original_row_count = self.data.shape[0]
        original_col_count = self.data.shape[1]

        unlabelled_row_count = self.dataset.unlabelled.row_count
        unlabelled_col_count = self.dataset.unlabelled.col_count

        labelled_row_count = self.dataset.labelled.row_count
        labelled_col_count = self.dataset.labelled.col_count

        # All of the data should be unlabelled
        self.assertEqual(original_row_count, unlabelled_row_count)
        # The labelled dataset should be empty
        self.assertEqual(0, labelled_row_count)

        # We expect one fewer column in our dataset (the index is not counted as a colummn)
        self.assertEqual(original_col_count - 1, unlabelled_col_count)
        self.assertEqual(unlabelled_col_count, labelled_col_count)

    def test_dataset_makes_copy_of_dataframe(self):
        self.assertIsNot(self.data, self.dataset.labelled._data)

    def test_index_is_set(self):
        without_index = self.data;

        without_index_col_count = without_index.shape[1]

        without_index_dataset_col_count = Dataset(without_index).unlabelled.col_count

        self.assertEqual(without_index_col_count - 1, without_index_dataset_col_count)

    def test_by_id_returns_single_row_for_single_id(self):
        id = self.dataset.unlabelled.index[0]
        found = self.dataset.unlabelled.by_id(id)
        self.assertIsInstance(found, pd.Series)

    def test_by_id_returns_dataframe_for_multi_id(self):
        ids = self.dataset.unlabelled.index[0:2]
        found = self.dataset.unlabelled.by_id(ids)
        self.assertEqual(found.shape[0], 2)


if __name__ == '__main__':
    unittest.main()