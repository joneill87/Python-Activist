from Activist.stoppingcriteria.stoppingcriterion import StoppingCriterion
from Activist.dataset import Dataset;
from Activist.domain.reporting import BatchSummary;


class StagnationCriterion(StoppingCriterion):

    def __init__(self, max_runs_without_improvement):
        self._max_runs_without_improvement = max_runs_without_improvement
        self._stagnant_run_count = 0
        self._correct_last_batch = 0

    def next_batch_size(self, data: Dataset, last_batch_summary: BatchSummary, batch_size: int) -> int:
        is_stagnant = last_batch_summary.correct <= self._correct_last_batch;

        # Keep track of the last number of times in a row we haven't seen an improvement
        if is_stagnant:
            self._stagnant_run_count = self._stagnant_run_count + 1
        else:
            self._stagnant_run_count = 0

        self._correct_last_batch = last_batch_summary.correct

        if self._stagnant_run_count == self._max_runs_without_improvement:
            return 0
        else:
            return batch_size
