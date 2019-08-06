from Activist.stoppingcriteria.stoppingcriterion import StoppingCriterion
from Activist.dataset import Dataset;
from Activist.domain.reporting import BatchEvaluation;


class LabelCountCriterion(StoppingCriterion):

    def __init__(self, max_labels):
        self._max_labels = max_labels

    def next_batch_size(self, data: Dataset, last_batch_evaluation: BatchEvaluation, batch_size: int) -> int:
        remaining_budget = self._max_labels - data.labelled.row_count
        return min(remaining_budget, batch_size)
