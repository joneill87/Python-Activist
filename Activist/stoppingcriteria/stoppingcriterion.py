from abc import abstractmethod
from Activist.dataset import Dataset
from Activist.domain.reporting import BatchSummary;


class StoppingCriterion:

	def init(self):
		pass

	@abstractmethod
	def next_batch_size(self, data: Dataset, last_batch_summary: BatchSummary, batch_size: int) -> int:
		pass
