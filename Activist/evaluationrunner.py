from stoppingcriterion import StoppingCriterion
from selectionstrategy import *
from model import Model, SKLearnKNN
from typing import Callable, Iterable, Any
from csv_types import CSVWriter
from pandas import DataFrame
from preprocessing import PreprocessingQueue
from preprocessors.standardScaler import StandardScaler
from dataset import Dataset

import pandas as pd
import numpy as np
import collections
import os
import csv


# Create a struct to hold our Batch Summary object. This simplifies writing to CSV
BatchSummary = collections.namedtuple('BatchSummary', ['batch', 'total', 'isLabelled', 'correct',
                                                       'incorrect', 'truePositive', 'trueNegative',
                                                       'falsePositive', 'falseNegative'])


class SSEvaluationRunner:
    """Used to carry out an evaluation of multiple selection strategies on one or more datasets"""

    def __init__(self,
                 data: Dataset,
                 seeder: Callable[[DataFrame, int], Iterable[Any]],
                 preprocessor: PreprocessingQueue,
                 num_seed_instances: int,
                 stopping_criterion: StoppingCriterion,
                 batch_size: int,
                 selection_strategy: SelectionStrategy,
                 model: Model,
                 output_dir: str,
                 file_identifier: str):

        self.data = data
        self.seeder = seeder
        self.preprocessor = preprocessor
        self.num_seed_instances = num_seed_instances
        self.stopping_criterion = stopping_criterion
        self.batch_size = batch_size
        self.model = model
        self.selection_strategy = selection_strategy

        # Get the paths to the output files.
        self.output_dir = output_dir
        self.detail_file_path = os.path.join(self.output_dir, "evaluation_" + file_identifier + "_detail.csv")
        self.summary_file_path = os.path.join(self.output_dir, "evaluation_" + file_identifier + "_summary.csv")

        # Create the directory if it doesn't exist
        directory = os.path.dirname(output_dir)
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Empty array to hold the IDs for rows which have already been queried
        self.queried_ids = []

    def add_query(self, ids: Iterable[Any]):
        # Currently we have all labels, even for the unlabelled dataset
        # We can just grab the labels from the unlabelled data and add
        # them
        rows_to_label = self.data.unlabelled.by_id(ids)
        self.data.add_labels(ids, rows_to_label.labels)
        queried_ids = self.queried_ids
        queried_ids.extend(ids)

        return self.data

    def run(self):

        batch_number = 0
        self.data = self.preprocessor.preprocess_data(self.data)

        seed_data = self.seeder(self.data, self.num_seed_instances)
        self.data = self.add_query(seed_data)
        unlabelled_obs = self.data.unlabelled.row_count

        evaluated = self.evaluate_batch(self.data, batch_number)

        with open(self.detail_file_path, "w") as f:
            evaluated.to_csv(f, header=True)

        summary = get_batch_summary(evaluated, batch_number)

        with open(self.summary_file_path, "w") as f:
            writer: CSVWriter = csv.writer(f)
            writer.writerow([header for header in summary._asdict().keys()])
            writer.writerow([val for val in summary])

        while unlabelled_obs > 0:
            next_query_size = min(unlabelled_obs, self.batch_size)
            next_query = self.selection_strategy.get_next_query(
                self.data.unlabelled,
                next_query_size)
            self.data = self.add_query(next_query)

            unlabelled_obs = self.data.unlabelled.row_count
            batch_number += 1
            evaluated = self.evaluate_batch(self.data, batch_number)
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

    def evaluate_batch(self, data: Dataset, batch_number):

        self.model.fit(data.labelled.features, data.labelled.labels)

        if data.unlabelled.row_count == 0:
            probabilities = np.array([])
        else:
            probabilities = self.model.predict_proba(data.unlabelled.features)

        train = data.labelled.copy()
        test = data.unlabelled.copy()

        test = test.assign(isLabelled=0)
        test = test.assign(confidence=[calculate_confidence(row) for row in probabilities])
        test = test.assign(prediction=[get_prediction(row) for row in probabilities])

        train = train.assign(isLabelled=1)
        train = train.assign(confidence=1)
        train = train.assign(prediction=train.label) # use previously supplied labels as prediction
        full = pd.concat([train, test])
        full = full.assign(batchNumber=batch_number)

        return full.filter(items=['batchNumber', 'id', 'isLabelled', 'confidence', 'prediction', 'label'])


def calculate_confidence(probabilities):
    return probabilities[np.argmax(probabilities)]


def get_prediction(probabilities):
    return np.argmax(probabilities)


def test_seeder(df, num_seed_instances):
    return df.unlabelled.head(num_seed_instances).index.values


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


def factorize_labels(labels):
    return {name: index for index, name in enumerate(labels.unique())}


if __name__ == "__main__":
    p_data = Dataset(pd.read_csv("data/iris.csv"))
    p_seeder = test_seeder
    p_preprocessor = PreprocessingQueue(StandardScaler())
    p_sc = None
    p_model = SKLearnKNN(3)
    p_selection_strategy = RandomSelectionStrategy(33)
    p_seed_count = 4
    p_batch_size = 2
    SSEvaluationRunner(p_data,
                       p_seeder,
                       p_preprocessor,
                       p_seed_count,
                       p_sc,
                       p_batch_size,
                       p_selection_strategy,
                       p_model,
                       "results/kneighbours/default/hedged_80/",
                       "k3").run()
