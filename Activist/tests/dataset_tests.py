import unittest

from Activist.dataset import Dataset

import pandas as pd


class TestDatasetMethods(unittest.TestCase):

    def setUp(self):
        self.data = pd.DataFrame([
            [1, 'one', 'ein', 'label_1'],
            [2, 'two', 'zwei', 'label_2'],
            [3, 'three', 'drei', 'label_3'],
            [4, 'four', 'vier', 'label_4']
        ], columns=['id', 'f1', 'f2', 'label'])

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
        self.assertEqual(1, found.row_count)

    def test_by_id_returns_dataframe_for_multi_id(self):
        ids = self.dataset.unlabelled.index[0:2]
        found = self.dataset.unlabelled.by_id(ids)
        self.assertEqual(2, found.shape[0])

    def test_add_labels(self):
        original_data_rows = self.data.shape[0]
        self.assertEqual(original_data_rows, 4)
        self.assertEqual(self.dataset.unlabelled.row_count, original_data_rows)
        id = self.dataset.unlabelled.index[0]
        label = 'new label'
        self.dataset.add_labels(id, label)
        self.assertEqual(self.dataset.labelled.row_count, 1)
        # we've added a label, we expect labelled to contain one row, and unlabelled to contain 1 fewer
        self.assertEqual(self.dataset.unlabelled.row_count, original_data_rows - 1)
        self.assertSequenceEqual([label], self.dataset.labelled.by_id(id).labels.tolist())

        ids = self.dataset.unlabelled.index[1:3]
        labels = ['new label 2', 'new label 3']
        self.assertEqual(len(ids), 2)

        self.dataset.add_labels(ids, labels)
        self.assertEqual(self.dataset.labelled.row_count, 3)
        self.assertEqual(self.dataset.unlabelled.row_count, original_data_rows - 3)
        print(self.dataset.labelled.by_id(ids).labels)
        self.assertSequenceEqual(labels, self.dataset.labelled.by_id(ids).labels.tolist())


    def test_by_id_returns_from_labelled_and_unlabelled(self):
        labelled_id = self.dataset.unlabelled.index[0]
        self.dataset.add_labels(labelled_id, 'new label')
        unlabelled_id = self.dataset.unlabelled.index[1]

        found = self.dataset.by_id([labelled_id, unlabelled_id])
        self.assertEqual(2, found.row_count)



if __name__ == '__main__':
    unittest.main()