try:
    import ujson
except ImportError:
    import json

class Writer:
    def __init__(self, dataset):
        self.dataset = dataset

    def __call__(self, path, num_splits=None, prefix=None):
        # TODO: shuffled executor for deterministic/even splits
        raise NotImplementedError

json = Writer
