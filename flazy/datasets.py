from abc import ABC, abstractmethod
from typing import List, Union

import random
import pandas as pd
from tqdm import tqdm
import numpy as np

from . import readers
from . import writers
from .utils import make_batch, close_iterator
from .executors import (
    Executor,
    BackgroundThreadExecutor,
    ThreadPoolExecutor,
    CurrentThreadExecutor,
    MultiProcessingExecutor
)


class Dataset(ABC):
    read = readers.LazyLoader()
    write: writers.LazyLoader

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls is Dataset and InMemoryDataset or cls)

    def __init__(self, *args, **kwargs):
        self.write = writers.LazyLoader(self)
        self._first = None
        self._shape = None
        self._types = None

    def map(self, mapper: callable, **executor_config) -> 'Dataset':
        return MappedDataset(self, mapper, **executor_config)

    def starmap(self, starmapper: callable, **executor_config) -> 'Dataset':
        return StarMappedDataset(self, starmapper, **executor_config)

    def flatmap(self, flatmapper: callable, **executor_config) -> 'Dataset':
        return FlatMappedDataset(self, flatmapper, **executor_config)

    def filter(self, predicate: callable, **executor_config) -> 'Dataset':
        return FilteredDataset(self, predicate, **executor_config)

    def transform(self, transformer, **executor_config) -> 'Dataset':
        return TransformedDataset(self, transformer, **executor_config)

    def foreach(self, function, **executor_config) -> None:
        """apply a function to each item"""
        for _ in self.map(lambda item: function(item) or True, **executor_config):
            pass

    def list(self, verbose: bool=False) -> list:
        """collect all entries as a list"""
        if not verbose:
            return list(self)
        else:
            return list(tqdm(self))

    def collect(self, default_dtype=np.float32, verbose: bool=False):
        """collect all entries into one object, stacking one-dimension-higher numpy arrays where applicable"""
        if not verbose:
            return make_batch(list(self), default_dtype=default_dtype)
        else:
            return make_batch(list(tqdm(self)), default_dtype=default_dtype)

    def first(self):
        """returns the first element"""
        try:
            if self._first is None:
                self._first = self.take(1).list()[0]
            return self._first
        except IndexError:
            raise ValueError('empty dataset')

    def shape(self, template=None):
        """returns the shape(s) of the items in this dataset"""
        is_root = template is None
        if is_root:
            if hasattr(self, '_shape') and self._shape is not None:
                return self._shape
            template = self.first()

        if isinstance(template, list):
            shape = [self.shape(item) for item in template]
        elif isinstance(template, np.ndarray):
            shape = template.shape
        elif isinstance(template, tuple):
            shape = tuple(self.shape(item) for item in template)
        elif isinstance(template, dict):
            shape = {key: self.shape(template[key]) for key in template}
        else:
            shape = tuple()

        if is_root:
            self._shape = shape

        return shape

    def types(self, template=None):
        """returns the type(s) of the items in this dataset"""
        is_root = template is None
        if is_root:
            if hasattr(self, '_types') and self._types is not None:
                return self._types
            template = self.first()

        if isinstance(template, list):
            types = [self.types(item) for item in template]
        elif isinstance(template, np.ndarray):
            types = template.dtype
        elif isinstance(template, tuple):
            types = tuple(self.types(item) for item in template)
        elif isinstance(template, dict):
            types = {key: self.types(template[key]) for key in template}
        elif isinstance(template, int):
            types = np.int64
        elif isinstance(template, float):
            types = np.float32
        else:
            types = None

        if is_root:
            self._types = types

        return types

    def tensorflow(self):
        """construct a tf.data.Dataset from this dataset"""
        import tensorflow as tf

        def np_to_tf(types):
            if isinstance(types, list):
                return [np_to_tf(item) for item in types]
            if isinstance(types, tuple):
                return tuple(np_to_tf(item) for item in types)
            if isinstance(types, dict):
                return {key: np_to_tf(types[key]) for key in types}
            if isinstance(types, float):
                return tf.float32
            if isinstance(types, int):
                return tf.int64
            if isinstance(types, str):
                return tf.string
            return tf.as_dtype(types)

        return tf.data.Dataset.from_generator(lambda: iter(self), np_to_tf(self.types()), self.shape())

    def take(self, count: int) -> 'Dataset':
        """returns a dataset containing the first few items only"""
        return SlicedDataset(self, 0, count)

    def skip(self, count: int) -> 'Dataset':
        """returns a dataset excluding the first few items"""
        return self.transform(lambda items: (item for i, item in enumerate(self) if i >= count), background=False)

    def __getitem__(self, item):
        """return a slice or a single item of this dataset"""
        if isinstance(item, slice):
            if item.start is None and item.stop is None and item.step is None:
                return self
            return SlicedDataset(self, item)
        elif item is ...:
            return self
        elif isinstance(item, int):
            return self.take(item).list()[-1]
        else:
            raise ValueError("Unsupported index:", item)

    def repeat(self, times: int=-1) -> 'Dataset':
        """loop over this dataset for a given number of cycles, or indefinitely by default"""
        return RepeatedDataset(self, times)

    def batch(self, size: int, default_dtype: np.dtype=np.float32) -> 'Dataset':
        """create a dataset consisting of mini-batches of the elements in this dataset"""
        return MiniBatchDataset(self, size, default_dtype=default_dtype)

    def cache(self, verbose=False) -> 'Dataset':
        """cache the content """
        return CachedDataset(self, verbose)

    def select(self, *keys, **executor_config):
        """for a dataset of dictionaries, select the fields with given keys"""
        return self.map(lambda row: {key: row[key] for key in keys}, **executor_config)

    def select_tuple(self, *keys, **executor_config):
        """for a dataset of dictionaries, select the fields with given keys as tuples"""
        return self.map(lambda row: tuple(row[key] for key in keys), **executor_config)

    def shuffle(self, buffer_size: int=None, seed=None):
        """shuffles the dataset using """
        return ShuffledDataset(self, buffer_size, seed)

    def sample(self, count: int, buffer_size: int=-1, seed=None):
        return self.shuffle(buffer_size, seed).take(count)

    @classmethod
    def concat(cls, datasets: List['Dataset']) -> 'Dataset':
        from .mux import SequentialMux
        return SequentialMux(datasets)

    @classmethod
    def roundrobin(cls, datasets: List['Dataset']) -> 'Dataset':
        from .mux import RoundRobinMux
        return RoundRobinMux(datasets)

    def __add__(self, other) -> 'Dataset':
        if isinstance(other, Dataset) or callable(other):
            from .mux import SequentialMux
            return SequentialMux([self, other])
        raise ValueError("unknown operand type: {}".format(type(other)))

    def __radd__(self, other) -> 'Dataset':
        return PrependedDataset(self, other)

    def executor(self, **executor_config) -> Executor:
        """returns an Executor instance to use in this dataset, according to the config"""
        if 'executor' in executor_config:
            executor = executor_config['executor']
            if isinstance(executor, Executor):
                return executor
        if 'background' in executor_config:
            if executor_config['background'] is True:
                return BackgroundThreadExecutor()
            else:
                return CurrentThreadExecutor()
        if 'num_threads' in executor_config:
            return ThreadPoolExecutor(int(executor_config['num_threads']))
        if 'num_processes' in executor_config:
            return MultiProcessingExecutor(int(executor_config['num_processes']))
        if len(executor_config) > 0:
            raise ValueError("Unknown executor_config:", executor_config.keys())
        try:
            return self._upstream()[0].executor()
        except IndexError:
            return CurrentThreadExecutor()

    @abstractmethod
    def _upstream(self) -> List['Dataset']:
        """subclasses should return a list of the upstream datasets that it depends on"""

    @abstractmethod
    def __iter__(self):
        """subclasses should implement how to iterate over this dataset"""

    def __repr__(self):
        try:
            shape = "shape {}".format(self.shape())
        except ValueError:
            shape = "unknown shape"
        if isinstance(shape, dict):
            shape = dict(sorted(shape.items()))  # dict() is sorted in Python >= 3.6
        return "(Dataset: {} of {})".format(type(self).__name__, shape)


class InMemoryDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()

        if (len(args) == 0) == (len(kwargs) == 0):
            raise ValueError('either args or kwargs must be given')

        if len(args) == 1:
            if isinstance(args[0], pd.DataFrame):
                # pandas dataframe to dataset of dicts
                self._items = args[0].to_dict('records')
            else:
                # dataset of single items
                self._items = args[0]
        elif len(args) > 0:
            # dataset of tuples:
            self._items = zip(*args)

        # dataset of dicts
        if len(kwargs) > 0:
            self._items = kwargs

    def _upstream(self):
        return []

    def __iter__(self):
        if isinstance(self._items, dict):
            keys = list(self._items.keys())
            values = [self._items[key] for key in keys]
            for record in zip(*values):
                yield {key: value for key, value in zip(keys, record)}
        else:
            items = callable(self._items) and self._items() or self._items
            iterator = iter(items)
            try:
                for item in iterator:
                    yield item
            finally:
                close_iterator(iterator)


class RepeatedDataset(Dataset):
    def __init__(self, upstream: Dataset, times: int=-1):
        super().__init__()
        self.upstream = upstream
        self.times = times

    def _upstream(self):
        return [self.upstream]

    def __iter__(self):
        for _ in self.times >= 0 and range(self.times) or iter(int, 1):
            iterator = iter(self.upstream)
            try:
                for item in iterator:
                    yield item
            finally:
                close_iterator(iterator)


class MiniBatchDataset(Dataset):
    def __init__(self, upstream: Dataset, size: int, default_dtype: np.dtype=None):
        super().__init__()
        self.upstream = upstream
        self.size = size
        self.default_dtype = default_dtype

    def _upstream(self):
        return [self.upstream]

    def __iter__(self):
        iterator = iter(self.upstream)
        try:
            while True:
                try:
                    yield make_batch([next(iterator) for _ in range(self.size)], self.default_dtype)
                except StopIteration:
                    return
        finally:
            close_iterator(iterator)


class TransformedDataset(Dataset):
    class TransformerGuard:
        def __init__(self, transformer: callable):
            self.transformer = transformer

        def __call__(self, upstream: Dataset):
            iterator = iter(upstream)
            try:
                for item in self.transformer(iterator):
                    yield item
            finally:
                close_iterator(iterator)

    def __init__(self, upstream: Dataset, transformer: callable, **executor_config):
        super().__init__()
        self.upstream = upstream
        self.transformer = transformer
        self.executor_config = executor_config

    def _upstream(self):
        return [self.upstream]

    def __iter__(self):
        return self.executor(**self.executor_config).execute(self.TransformerGuard(self.transformer), self.upstream)


class MappedDataset(TransformedDataset):
    def __init__(self, upstream: Dataset, mapper: callable, **executor_config):
        super().__init__(upstream, lambda items: (mapper(x) for x in items), **executor_config)


class StarMappedDataset(TransformedDataset):
    def __init__(self, upstream: Dataset, starmapper: callable, **executor_config):
        super().__init__(upstream, lambda items: (isinstance(x, dict) and starmapper(**x) or starmapper(*x) for x in items), **executor_config)


class FilteredDataset(TransformedDataset):
    def __init__(self, upstream: Dataset, predicate: callable, **executor_config):
        super().__init__(upstream, lambda items: (x for x in items if predicate(x)), **executor_config)


class FlatMappedDataset(TransformedDataset):
    def __init__(self, upstream: Dataset, flatmapper: callable, **executor_config):
        super().__init__(upstream, lambda items: (y for x in items for y in flatmapper(x)), **executor_config)


class SlicedDataset(Dataset):
    def __init__(self, upstream: Dataset, start: Union[int, slice], stop: int=None, step: int=1):
        super().__init__()
        self.upstream = upstream
        if isinstance(start, slice):
            self.start, self.stop, self.step = start.start or 0, start.stop or -1, start.step or 1
        else:
            self.start, self.stop, self.step = start or 0, stop or -1, step or 1

    def _upstream(self) -> List['Dataset']:
        return [self.upstream]

    def __iter__(self):
        count = 0
        iterator = iter(self.upstream)
        try:
            for item in iterator:
                if count == self.stop:
                    break
                d = count - self.start
                if d >= 0 and d % self.step == 0:
                    yield item
                count += 1
        finally:
            close_iterator(iterator)


class CachedDataset(Dataset):
    def __init__(self, upstream: Dataset, verbose: bool):
        super().__init__()
        self.upstream = upstream
        self.verbose = verbose
        self.cache = upstream.list(verbose=verbose)

    def _upstream(self):
        return self.upstream

    def __iter__(self):
        return iter(self.cache)


class ShuffledDataset(Dataset):
    def __init__(self, upstream, buffer_size=None, seed=None):
        super().__init__()
        self.upstream = upstream
        self.buffer_size = buffer_size
        self.random = random.Random(seed)

    def _upstream(self) -> List['Dataset']:
        return [self.upstream]

    def __iter__(self):
        iterator = iter(self.upstream)
        try:
            # fill the buffer
            if self.buffer_size is not None:
                buffer_size = self.buffer_size
                buffer = [item for _, item in zip(range(buffer_size), iterator)]
            else:
                buffer = list(iterator)
                buffer_size = len(buffer)
            self.random.shuffle(buffer)

            # sample one from the buffer and replace with a new one pulled from the iterator
            for item in iterator:
                i = self.random.randrange(buffer_size)
                yield buffer[i]
                buffer[i] = item

            # drain any remaining items
            for item in buffer:
                if item is not None:
                    yield item
        finally:
            close_iterator(iterator)


class PrependedDataset(Dataset):
    def __init__(self, upstream, other):
        super().__init__()
        self.upstream = upstream
        self.other = other

    def _upstream(self) -> List['Dataset']:
        return [self.upstream]

    def __iter__(self):
        yield self.other
        iterator = iter(self.upstream)
        try:
            for item in self.upstream:
                yield item
        finally:
            close_iterator(iterator)
