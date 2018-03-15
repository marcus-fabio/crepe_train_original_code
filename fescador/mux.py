from random import Random
from typing import List

from .datasets import Dataset
from .executors import Executor, CurrentThreadExecutor


class Mux(Dataset):
    def __init__(self, datasets: List[Dataset]):
        self.datasets = datasets

    def _upstream(self) -> List[Dataset]:
        return self.datasets

    def executor(self, **kwargs) -> Executor:
        return CurrentThreadExecutor()


class SequentialMux(Mux):
    def __iter__(self):
        for dataset in self.datasets:
            for item in dataset:
                yield item


class RoundRobinMux(Mux):
    def __iter__(self):
        iterators = list(map(iter, self.datasets))
        while len(iterators) > 0:
            for iterator in iterators:
                try:
                    yield next(iterator)
                except StopIteration:
                    iterators.remove(iterator)
