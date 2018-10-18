from stoppingcriterion import StoppingCriterion
from selectionstrategy import *
from model import Model, SKLearnKNN
from typing import Callable, Iterable, Any
from csv_types import CSVWriter
from pandas import DataFrame

from sklearn import preprocessing

import pandas as pd
import numpy as np
import collections
import os
import csv

BatchSummary = collections.namedtuple('BatchSummary', ['batch', 'total', 'isLabelled', 'correct',
                                                       'incorrect', 'truePositive', 'trueNegative',
                                                       'falsePositive', 'falseNegative'])


class SSEvaluationRunner:
    """Used to carry out an evaluation of multiple selection strategies on one or more datasets"""

    def __init__(self,
                 data: DataFrame,
                 seeder: Callable[[DataFrame, int], Iterable[Any]],
                 num_seed_instances: int,
                 stopping_criterion: StoppingCriterion,
                 batch_size: int,
                 selection_strategy: SelectionStrategy,
                 model: Model,
                 output_dir: str,
                 file_identifier: str):

        self.data = data
        self.seeder = seeder
        self.num_seed_instances = num_seed_instances
        self.stopping_criterion = stopping_criterion
        self.batch_size = batch_size
        self.model = model
        self.selection_strategy = selection_strategy
        self.output_dir = output_dir
        self.detail_file_path = os.path.join(self.output_dir, "evaluation_" + file_identifier + "_detail.csv")
        self.summary_file_path = os.path.join(self.output_dir, "evaluation_" + file_identifier + "_summary.csv")

        directory = os.path.dirname(output_dir)
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.unlabelled = data
        self.labelled = DataFrame().reindex_like(data).iloc[0:0]  # create an empty copy of the unlabelled data frame

        self.ids = self.unlabelled.columns[1]
        self.feature_columns = self.unlabelled.columns[1:-2]
        self.label_column = self.unlabelled.columns[-1]
        self.queried_ids = []

    def add_query(self, ids: Iterable[Any]):
        labelled = self.labelled
        unlabelled = self.unlabelled
        queried_ids = self.queried_ids

        queried_ids.extend(ids)
        new_labelled = unlabelled.loc[unlabelled.id.isin(queried_ids)]
        labelled = pd.concat([labelled, new_labelled])
        unlabelled = unlabelled.loc[~unlabelled.id.isin(queried_ids)]
        return labelled, unlabelled

    def preprocess_data(self, data):
        data[self.feature_columns] = preprocessing.StandardScaler().fit_transform(data[self.feature_columns])
        return data

    def run(self):

        batch_number = 0
        self.data = self.preprocess_data(self.data)
        seed_data = self.seeder(self.data, self.num_seed_instances)
        train, test = self.add_query(seed_data)
        unlabelled_obs = test.shape[0]

        evaluated = self.evaluate_batch(train, test, batch_number)

        with open(self.detail_file_path, "w") as f:
            evaluated.to_csv(f, header=True)

        summary = get_batch_summary(evaluated, batch_number)

        with open(self.summary_file_path, "w") as f:
            writer: CSVWriter = csv.writer(f)
            writer.writerow([header for header in summary._asdict().keys()])
            writer.writerow([val for val in summary])

        while unlabelled_obs > 0:
            next_query_size = min(unlabelled_obs, self.batch_size)
            l_j = evaluated.loc[evaluated.isLabelled == 1]
            l_u = evaluated.loc[evaluated.isLabelled == 0]
            next_query = self.selection_strategy.get_next_query(
                evaluated.loc[evaluated.isLabelled == 0],
                next_query_size)
            train, test = self.add_query(next_query)

            unlabelled_obs = test.shape[0]
            batch_number += 1
            evaluated = self.evaluate_batch(train, test, batch_number)
            with open(self.detail_file_path, "a") as f:
                evaluated.to_csv(f, header=False)

            summary = get_batch_summary(evaluated, batch_number)

            with open(self.summary_file_path, "a") as f:
                writer = csv.writer(f)
                writer.writerow([val for val in summary])
                percent_correct = summary.correct / summary.total * 100
                print("Evaluated Batch Number {} unlabelled remaining: {} accuracy {:4f}%".format(batch_number,
                                                                                                  unlabelled_obs,
                                                                                                  percent_correct))

    def evaluate_batch(self, train, test, batch_number):
        self.model.fit(train[self.feature_columns], train[self.label_column])
        if len(test) == 0:
            probabilities = np.array([])
        else:
            probabilities = self.model.predict_proba(test[self.feature_columns])

        test = test.assign(isLabelled=0)
        test = test.assign(confidence=[calculate_confidence(row) for row in probabilities])
        test = test.assign(prediction=[get_prediction(row) for row in probabilities])

        train = train.assign(isLabelled=1)
        train = train.assign(confidence=1)
        train = train.assign(prediction=train.label)
        full = pd.concat([train, test])
        full = full.assign(batchNumber=batch_number)

        return full.filter(items=['batchNumber', 'id', 'isLabelled', 'confidence', 'prediction', 'label'])


def calculate_confidence(probabilities):
    return probabilities[np.argmax(probabilities)]


def get_prediction(probabilities):
    return np.argmax(probabilities)


def test_seeder(df, num_seed_instances):
    return df.head(num_seed_instances)['id']


def get_batch_summary(evaluated, batch_number):
        total = evaluated.shape[0]
        label_count = evaluated.isLabelled.sum()
        correct = evaluated.query('prediction == label').shape[0]
        incorrect = evaluated.query('prediction != label').shape[0]
        true_positive = evaluated.query('prediction == label & prediction == 1').shape[0]
        true_negative = evaluated.query('prediction == label & prediction == 0').shape[0]
        false_positive = evaluated.query('prediction != label & prediction == 1').shape[0]
        false_negative = evaluated.query('prediction != label & prediction == 0').shape[0]
        return BatchSummary(batch_number, total, label_count, correct, incorrect, true_positive, true_negative,
                            false_positive, false_negative)


if __name__ == "__main__":
    p_data = pd.read_csv("data/labelled_tree_canopy_data_scaled.csv")
    p_seeder = test_seeder
    p_sc = None
    p_model = SKLearnKNN(3)
    p_selection_strategy = HedgedSelectionStrategy(0.2)
    p_seed_count = 10
    p_batch_size = 10
    SSEvaluationRunner(p_data,
                       p_seeder,
                       p_seed_count,
                       p_sc,
                       p_batch_size,
                       p_selection_strategy,
                       p_model,
                       "results/kneighbours/default/hedged_80/",
                       "k3").run()
