from abc import abstractmethod
from typing import Iterable, Any


class Dataset:

	@abstractmethod
	def get_ids(self) -> Iterable[Any]:
		pass

	@abstractmethod
	def get_features(self, ids: Iterable[Any]) -> Iterable[Iterable[Any]]:
		pass

	@abstractmethod
	def get_labels(self, ids: Iterable[Any]) -> Iterable[Any]:
		pass
