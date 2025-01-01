import os
import argparse
from typing import List
from datetime import datetime

from keras import Model, models
from keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    TensorBoard,
    CSVLogger,
    Callback
)

import models as crepe_models  # noqa

parser = argparse.ArgumentParser('CREPE', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('model', nargs='?', default='crepe',
                    help='name of the model')
parser.add_argument('experiment_name', nargs='?', default=datetime.now().strftime('%Y-%m-%dT%H_%M_%S'),
                    help='a unique identifier string for this run')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='an option to disable data augmentation')
parser.add_argument('--optimizer', default='adam',
                    help='the name of Keras optimizer to use')
parser.add_argument('--batch-size', default=32, type=int,
                    help='the mini-batch size')
parser.add_argument('--validation-take', default=4000, type=int,
                    help='how many examples to take from each validation dataset')
parser.add_argument('--model-capacity', default=32, type=int,
                    help='a multiplier to adjust the model capacity')
parser.add_argument('--load-model', default=None,
                    help='when specified, the full model will be loaded from this path')
parser.add_argument('--load-model-weights', default=None,
                    help='when specified, the model weights will be loaded from this path')
parser.add_argument('--save-model', default='model.h5',
                    help='path to save the model on each epoch')
parser.add_argument('--save-model-weights', default='model.h5',
                    help='path to save the model weights on each epoch; supersedes --save-model')
parser.add_argument('--epochs', default=300, type=int,
                    help='number of epochs to train')
parser.add_argument('--steps-per-epoch', default=500, type=int,
                    help='number of steps in a batch')
parser.add_argument('--tensorboard', default=False, action='store_true',
                    help='when enabled, tensorboard data will be saved under the log directory')

options = vars(parser.parse_args())
log_dir = os.path.join('experiments', options['experiment_name'])
os.makedirs(log_dir, exist_ok=True)


def log_path(*components):
    return os.path.join(log_dir, *components)


def build_model() -> Model:
    """returns the Keras model according to the options"""
    if options['load_model']:
        return models.load_model(options['load_model'])
    else:
        model: Model = getattr(crepe_models, options['model'])(**options)
        if options['load_model_weights']:
            model.load_weights(options['load_model_weights'])
        return model


def get_default_callbacks(custom_callback) -> List[Callback]:
    """returns a list of callbacks that are used by default"""
    result: List[Callback] = [
        CSVLogger(log_path('learning-curve.tsv'), separator='\t'),
    ]

    if options['save_model_weights']:
        result.append(ModelCheckpoint(log_path(options['save_model_weights']),
                                      save_best_only=True,
                                      save_weights_only=True,
                                      verbose=1))
    elif options['save_model']:
        result.append(ModelCheckpoint(log_path(options['save_model']), save_best_only=True))

    if options['tensorboard']:
        result.append(TensorBoard(log_path('tensorboard')))

    result.append(EarlyStopping(monitor='val_loss', patience=32, verbose=1, mode='min'))
    result.append(custom_callback)

    return result
