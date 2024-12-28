import os
from tfrecord_lite import decode_example
from typing import Optional, List

import tensorflow as tf

from .. import Dataset
from ..utils import close_iterator


def read_records(path: str, keys: List[str], options: tf.io.TFRecordOptions):
    if os.path.isdir(path):
        files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.tfrecord')]
    elif os.path.isfile(path):
        files = [path]
    else:
        raise ValueError('Could not open {}'.format(path))

    for file in files:
        # iterator = tf.io.tf_record_iterator(file, options)
        dataset = tf.data.TFRecordDataset(file, compression_type=options.compression_type if options else None)

        try:
            # for record in iterator:
            for raw_record in dataset:
                # yield decode_example(record, keys or [])
                yield decode_example(raw_record.numpy(), keys or [])
        finally:
            # close_iterator(iterator)
            close_iterator(dataset)


def tfrecord(*paths: str, keys: List[str]=None, compression: Optional[str]=None, **executor_config):
    options = None
    if compression and compression.lower() == 'gzip':
        options = tf.io.TFRecordOptions(compression_type='GZIP')
    elif compression and compression.lower() == 'zlib':
        options = tf.io.TFRecordOptions(compression_type='ZLIB')
    return Dataset(paths).flatmap(lambda path: read_records(path, keys, options), **executor_config)
